"""
Build an OpenFold3 training set from the OpenBind EV-A71 2A fragment screen.

    python scripts/prepare_finetune.py                    # about 30 s on CPU
    python scripts/prepare_finetune.py --limit 4 --check  # quick test: 4 complexes, then load them with OF3

Selection: fragment-screen complexes with a PoseBusters-valid reference pose, no
suspected crystallographic artefact and no covalent ligand, one binding event per
compound (66 complexes). Follow-on compounds are never trained on, so the
benchmark can measure how the fine-tune generalizes beyond the fragments.

Output, in the layout OF3's WeightedPDBDataset reads (finetune_ray_train.py points at it):

    $AIDDD_SHARED/finetune/ev71-2a/
        structures/<complex>/<complex>.npz    cleaned AtomArray: one protein chain, Zn, ligand
        reference_mols/<id>.sdf               one per residue type, Zn and ligand (annotated SDF)
        alignments/EV-A71_2A/cfdb_uniref30.a3m  the release MSA, shared by every complex
        training_cache.json                   chains, interfaces, clusters (sampling weights)

Why not OF3's own mmCIF preprocessing: the release ships REFMAC PDB files whose
ligand residue (LIG) is not in the Chemical Component Dictionary, so the standard
cleanup would delete it. Instead each complex is rebuilt from the sequence: every
residue of the 150 aa construct (coordinates from the crystal, NaN where unresolved,
so residue numbers line up with the MSA columns), the structural zinc, and the
ligand from its reference SDF with bond orders checked against the SMILES.
Waters, DMSO and sulfate are dropped.
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np
import pandas as pd
from biotite.interface.rdkit import from_mol
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from openfold3.core.data.io.structure.atom_array import write_atomarray_to_npz
from openfold3.core.data.pipelines.preprocessing.structure import _process_component
from openfold3.core.data.primitives.structure.component import mol_from_atomarray
from openfold3.core.data.primitives.structure.interface import get_interface_chain_id_pairs
from openfold3.core.data.resources.residues import PROTEIN_RESTYPE_1TO3, MoleculeType

RDLogger.DisableLog("rdApp.*")

SHARED = Path(os.environ.get("AIDDD_SHARED", "/mnt/user_storage/aiddd"))
RELEASE = SHARED / "raw" / "OpenBind_EV-A71_2A"
MSA_ID = "EV-A71_2A"            # alignment representative shared by every complex
MSA_NAME = "cfdb_uniref30"      # must be a key of OF3's MSASettings.max_seq_counts
PROTEIN, ZINC, LIGAND = "1", "2", "3"   # chain ids in the training AtomArrays
ANNOTATIONS = ["chain_id", "res_id", "ins_code", "res_name", "hetero", "atom_name",
               "element", "charge", "occupancy"]


def select_complexes(meta: pd.DataFrame) -> pd.DataFrame:
    """Fragment screen, clean reference pose, one binding event per compound."""
    ok = (meta.fragment_screen & meta.pb_valid_ref & ~meta.suspected_artefact
          & ~meta.covalent)
    return (meta[ok].sort_values("complex_name")
            .drop_duplicates("compound_group").reset_index(drop=True))


def blank(n: int) -> struc.AtomArray:
    """Empty AtomArray with the annotation set every segment shares."""
    arr = struc.AtomArray(n)
    arr.set_annotation("charge", np.zeros(n, int))
    arr.set_annotation("occupancy", np.ones(n))
    return arr


def from_template(template: struc.AtomArray, coord: np.ndarray, chain: str) -> struc.AtomArray:
    arr = blank(len(template))
    for name in ANNOTATIONS[:-1]:
        arr.set_annotation(name, template.get_annotation(name))
    arr.coord = coord
    arr.chain_id[:] = chain
    arr.bonds = template.bonds
    return arr


def build_protein(seq: str, crystal: struc.AtomArray) -> struc.AtomArray:
    """Full-length chain from CCD residues; crystal coordinates where resolved."""
    resolved = {(r, a): xyz for r, a, xyz in zip(crystal.res_id, crystal.atom_name, crystal.coord)}
    residues = []
    for res_id, aa in enumerate(seq, start=1):
        res = struc.info.residue(PROTEIN_RESTYPE_1TO3[aa])
        res = res[(res.element != "H") & (res.atom_name != "OXT")]   # OF3 drops terminal OXT
        res.res_id[:] = res_id
        coord = np.array([resolved.get((res_id, a), [np.nan] * 3) for a in res.atom_name])
        residues.append(from_template(res, coord, PROTEIN))
    chain = struc.concatenate(residues)
    chain.bonds = struc.connect_via_residue_names(chain)
    chain.occupancy[np.isnan(chain.coord).any(axis=1)] = 0.0
    return chain


def load_ligand(sdf: Path, smiles: str) -> Chem.Mol:
    """Crystal pose with bond orders taken from the SMILES (falls back to the SDF's own)."""
    mol = Chem.MolFromMolFile(str(sdf), removeHs=True)
    ref = Chem.MolFromSmiles(smiles)
    if mol is None or ref is None:
        raise ValueError("ligand does not parse")
    if mol.GetNumHeavyAtoms() != ref.GetNumHeavyAtoms():
        raise ValueError(f"partial ligand: {mol.GetNumHeavyAtoms()} of {ref.GetNumHeavyAtoms()} atoms")
    try:
        mol = AllChem.AssignBondOrdersFromTemplate(ref, mol)
    except ValueError:
        pass
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


def build_ligand(mol: Chem.Mol, res_name: str) -> struc.AtomArray:
    lig = from_mol(mol, conformer_id=0, add_hydrogen=False)
    lig.res_name[:] = res_name
    lig.res_id[:] = 1
    lig.hetero[:] = True
    lig.atom_name = struc.create_atom_names(lig)
    lig.element = np.char.upper(lig.element)
    return from_template(lig, lig.coord, LIGAND)


def resolution(pdb_path: Path) -> float:
    for line in open(pdb_path):
        if "RESOLUTION RANGE HIGH" in line:
            return float(line.split(":")[1])
    return 2.0


def release_date(pdb_path: Path) -> str:
    try:
        return datetime.strptime(open(pdb_path).readline()[50:59].strip(), "%d-%b-%y").strftime("%Y-%m-%d")
    except ValueError:
        return "2026-01-01"


def build_complex(row, seq: str, lig_id: str) -> tuple[struc.AtomArray, Chem.Mol]:
    """One training AtomArray: the protein copy the ligand binds, its zinc, the ligand."""
    folder = RELEASE / "structures" / row.compound_group / row.complex_name
    crystal = pdb.PDBFile.read(folder / f"{row.complex_name}_complex_ref.pdb").get_structure(
        model=1, altloc="occupancy", extra_fields=["occupancy"])
    crystal = crystal[crystal.element != "H"]
    mol = load_ligand(folder / f"{row.complex_name}_ligand_ref.sdf", row.smiles)
    lig_xyz = mol.GetConformer().GetPositions()

    # Crystals hold one or two protein copies: keep the one the ligand contacts
    protein = crystal[struc.filter_amino_acids(crystal) & ~crystal.hetero]
    dist = {c: np.min(np.linalg.norm(protein.coord[protein.chain_id == c][:, None] - lig_xyz, axis=-1))
            for c in np.unique(protein.chain_id)}
    chain = protein[protein.chain_id == min(dist, key=dist.get)]
    parts = [build_protein(seq, chain)]

    # Structural zinc of that copy (coordinated by Cys/His, within 3 A of the chain)
    zinc = crystal[crystal.res_name == "ZN"]
    if len(zinc):
        d = np.min(np.linalg.norm(zinc.coord[:, None] - chain.coord, axis=-1), axis=1)
        if d.min() < 3.0:
            zn = struc.info.residue("ZN")
            parts.append(from_template(zn, zinc.coord[[d.argmin()]], ZINC))

    parts.append(build_ligand(mol, lig_id))
    atoms = struc.concatenate(parts)

    # Annotations OF3's mmCIF preprocessing writes
    chains = {PROTEIN: (1, MoleculeType.PROTEIN, "A"), ZINC: (2, MoleculeType.LIGAND, "B"),
              LIGAND: (3, MoleculeType.LIGAND, "C")}
    atoms.set_annotation("entity_id", np.array([chains[c][0] for c in atoms.chain_id]))
    atoms.set_annotation("label_entity_id", atoms.entity_id.copy())
    atoms.set_annotation("molecule_type_id", np.array([int(chains[c][1]) for c in atoms.chain_id]))
    atoms.set_annotation("label_asym_id", np.array([chains[c][2] for c in atoms.chain_id]))
    atoms.set_annotation("auth_asym_id", atoms.label_asym_id.copy())
    atoms.set_annotation("label_seq_id", np.where(atoms.chain_id == PROTEIN, atoms.res_id.astype(str), "."))
    atoms.set_annotation("auth_seq_id", atoms.res_id.copy())
    for prefix in ("label", "auth"):
        atoms.set_annotation(f"{prefix}_comp_id", atoms.res_name.copy())
        atoms.set_annotation(f"{prefix}_atom_id", atoms.atom_name.copy())
    atoms.set_annotation("sym_id", np.zeros(len(atoms), int))
    return atoms, mol


def cache_entry(atoms: struc.AtomArray, row, lig_id: str, n_total: int, pdb_path: Path) -> dict:
    """Training-cache record. Clusters: the protein and zinc are shared by every complex
    (one big cluster each), every compound is its own cluster, so OF3's sampler favours
    the protein-ligand interface (AF3 SI 2.5.1)."""
    chains = {}
    for cid in np.unique(atoms.chain_id):
        is_prot = cid == PROTEIN
        ref = None if is_prot else ("ZN" if cid == ZINC else lig_id)
        cluster = {PROTEIN: ("ev71_2a", n_total), ZINC: ("zn", n_total),
                   LIGAND: (row.compound_group, 1)}[cid]
        chains[cid] = {
            "label_asym_id": atoms.label_asym_id[atoms.chain_id == cid][0],
            "auth_asym_id": atoms.label_asym_id[atoms.chain_id == cid][0],
            "entity_id": int(atoms.entity_id[atoms.chain_id == cid][0]),
            "molecule_type": "PROTEIN" if is_prot else "LIGAND",
            "reference_mol_id": ref,
            "alignment_representative_id": MSA_ID if is_prot else None,
            "template_ids": [] if is_prot else None,
            "cluster_id": cluster[0], "cluster_size": cluster[1]}
    interfaces = {}
    for a, b in get_interface_chain_id_pairs(atoms, distance_threshold=5.0):
        a, b = str(a), str(b)
        size = 1 if LIGAND in (a, b) else n_total
        interfaces[f"{a}_{b}"] = {"cluster_id": f"{chains[a]['cluster_id']}_{chains[b]['cluster_id']}",
                                  "cluster_size": size}
    return {"release_date": release_date(pdb_path), "resolution": resolution(pdb_path),
            "chains": chains, "interfaces": interfaces}


def ref_mol_record(meta: dict) -> dict:
    keys = ("conformer_gen_strategy", "fallback_conformer_pdb_id", "canonical_smiles",
            "fallback_conformer_source")
    return {**{k: meta.get(k) for k in keys}, "set_fallback_to_nan": False}


def check(out: Path, n: int = 4) -> None:
    """Load the set through OF3's own WeightedPDBDataset and featurize a few samples."""
    from openfold3.core.data.framework.single_datasets.pdb import WeightedPDBDataset
    from openfold3.projects.of3_all_atom.config.dataset_configs import TrainingDatasetSpec

    spec = TrainingDatasetSpec.model_validate({
        "name": "openbind-ev71", "dataset_class": "WeightedPDBDataset", "mode": "train",
        "weight": 1.0, "config": {"crop": {"token_crop": {"token_budget": 192}},
                                  "dataset_paths": dataset_paths(out)}})
    ds = WeightedPDBDataset(spec.config)
    print(f"WeightedPDBDataset: {len(ds)} datapoints (chains + interfaces)")
    for i in np.random.default_rng(0).choice(len(ds), size=min(n, len(ds)), replace=False):
        f = ds[int(i)]
        resolved = f["ground_truth"]["atom_resolved_mask"] if "ground_truth" in f else f.get("atom_resolved_mask")
        print(f"  {f['pdb_id']:<18} {f['preferred_chain_or_interface']:<12} "
              f"tokens={int(f['token_mask'].sum())} msa={tuple(f['msa'].shape)}"
              + (f" resolved_atoms={int(resolved.sum())}" if resolved is not None else ""))


def dataset_paths(out: Path) -> dict:
    """The `dataset_paths` block of the OF3 training config for this set."""
    return {"dataset_cache_file": str(out / "training_cache.json"),
            "alignments_directory": str(out / "alignments"),
            "target_structures_directory": str(out / "structures"),
            "target_structure_file_format": "npz",
            "reference_molecule_directory": str(out / "reference_mols")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=SHARED / "finetune" / "ev71-2a")
    ap.add_argument("--limit", type=int, default=None, help="first N complexes only (testing)")
    ap.add_argument("--check", action="store_true", help="load a few samples with OF3 afterwards")
    args = ap.parse_args()

    seq = "".join(open(RELEASE / "EV-A71_2A.fasta").read().split("\n")[1:]).strip()
    rows = select_complexes(pd.read_csv(RELEASE / "EV-A71_2A_metadata.csv"))
    if args.limit:
        rows = rows.head(args.limit)
    for sub in ("structures", "reference_mols", f"alignments/{MSA_ID}"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    # The release MSA, query first, reused for every complex
    msa = RELEASE / "EV-A71_2A_cfdb_msa.a3m"
    assert open(msa).read().split("\n")[1].strip() == seq, "MSA query does not match the FASTA"
    shutil.copy(msa, args.out / "alignments" / MSA_ID / f"{MSA_NAME}.a3m")

    structures, ref_mols = {}, {}
    for i, row in enumerate(rows.itertuples()):
        lig_id = f"OB{i:03d}"   # 5-character residue name, not a CCD code
        pdb_path = RELEASE / "structures" / row.compound_group / row.complex_name / f"{row.complex_name}_complex_ref.pdb"
        try:
            atoms, _ = build_complex(row, seq, lig_id)
        except Exception as e:   # report and skip a broken entry
            print(f"skip {row.complex_name}: {e}")
            continue
        (args.out / "structures" / row.complex_name).mkdir(exist_ok=True)
        write_atomarray_to_npz(atoms, args.out / "structures" / row.complex_name / f"{row.complex_name}.npz")
        structures[row.complex_name] = cache_entry(atoms, row, lig_id, len(rows), pdb_path)
        lig = atoms[atoms.chain_id == LIGAND]
        ref_mols[lig_id] = ref_mol_record(_process_component(lig_id, mol_from_atomarray(lig), 1,
                                                             args.out / "reference_mols"))

    # Reference molecules for the residue types and zinc, from the CCD copy in biotite
    for code in sorted(set(PROTEIN_RESTYPE_1TO3[a] for a in seq)) + ["ZN"]:
        res = struc.info.residue(code)
        mol = mol_from_atomarray(res[res.element != "H"])
        ref_mols[code] = ref_mol_record(_process_component(code, mol, 1, args.out / "reference_mols"))

    cache = {"_type": "ClusteredDatasetCache", "name": "openbind-ev71-2a-fragments",
             "structure_data": structures, "reference_molecule_data": ref_mols}
    (args.out / "training_cache.json").write_text(json.dumps(cache, indent=1))
    print(f"{len(structures)} complexes, {len(ref_mols)} reference molecules -> {args.out}")

    if args.check:
        check(args.out)


if __name__ == "__main__":
    main()
