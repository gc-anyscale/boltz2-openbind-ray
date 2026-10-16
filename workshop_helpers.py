"""
Display helpers for workshop.ipynb: an inline 3D viewer and result tables.

The Ray code lives in the notebook itself. This module only draws things, so the
notebook cells stay about Ray and the science.
"""

import html
import json
import os
import uuid
import warnings
from pathlib import Path

import pandas as pd
from IPython.display import HTML, display

from aiddd_pipeline import SHARED, STAGES, TARGETS, pose_in_crystal_frame

USER = os.environ.get("ANYSCALE_USERNAME") or os.environ.get("USER") or "attendee"
MINE = SHARED / "workshop" / USER          # everything this attendee writes goes here

MODEL_NAMES = {"boltz2": "Boltz-2", "openbind0": "OpenBind-0", "of3ft": "OF3 fine-tuned (EV-A71 2A)"}
COLORS = {"crystal": "#2b2b2b", "boltz2": "#1F5A96", "openbind0": "#E1A33B", "of3ft": "#B3412E"}
TARGET_KEYS = ["ev71-2a", "sars2-mpro", "mers-mpro"]
warnings.filterwarnings("ignore", category=FutureWarning)
_3DMOL = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"
# Shown in the viewer box until 3Dmol.js draws over it. A notebook reopened without being trusted
# keeps the saved HTML but does not run its script, so the 3D view only appears after running the cell again.
_FALLBACK = ('<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;'
             'color:#8a857c;font:13px system-ui,sans-serif">3D view not loaded: run this cell '
             '(or trust the notebook) to show it.</div>')


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------
def _viewer(models, legend="", height=420, zoom=None, zoom_out=0.85):
    """models: [{"text", "fmt", "styles": [(selection, style), ...]}]. Loads 3Dmol.js once."""
    div = "v" + uuid.uuid4().hex[:10]
    payload = json.dumps(models)
    js = f"""
(function() {{
  function draw() {{
    var el = document.getElementById("{div}");
    el.innerHTML = "";
    var v = $3Dmol.createViewer(el, {{backgroundColor: "white"}});
    var models = {payload};
    models.forEach(function(m, i) {{
      v.addModel(m.text, m.fmt);
      v.setStyle({{model: i}}, {{}});
      m.styles.forEach(function(s) {{ s[0].model = i; v.addStyle(s[0], s[1]); }});
    }});
    var zoom = {json.dumps(zoom or {})};
    if (zoom === "ligands")   // the reference ligand (model 1) and every predicted ligand (chain B)
      zoom = {{predicate: function(a) {{ return a.model === 1 || (a.model > 1 && a.chain === "B"); }}}};
    v.zoomTo(zoom); v.zoom({zoom_out}); v.render();
  }}
  if (window.$3Dmol) {{ draw(); return; }}
  var s = document.createElement("script"); s.src = "{_3DMOL}"; s.onload = draw;
  s.onerror = function() {{ document.getElementById("{div}").firstChild.textContent =
    "Could not download 3Dmol.js from cdnjs.cloudflare.com. Check the internet access, then run this cell again."; }};
  document.head.appendChild(s);
}})();"""
    display(HTML(f'<div style="font:13px system-ui,sans-serif;margin:4px 0">{legend}</div>'
                 f'<div id="{div}" style="width:100%;max-width:820px;height:{height}px;position:relative;'
                 f'border:1px solid #e5e5e5;border-radius:6px">{_FALLBACK}</div><script>{js}</script>'))


def _chip(color, text):
    return (f'<span style="display:inline-block;margin-right:14px"><span style="display:inline-block;width:10px;'
            f'height:10px;border-radius:2px;background:{color};margin-right:5px"></span>{html.escape(text)}</span>')


def _stick(color):
    return {"stick": {"color": color, "radius": 0.22}}


def show_complex(row, models=("boltz2", "openbind0"), height=420):
    """One crystal complex: the crystal ligand (dark) and each model's predicted ligand,
    moved onto the crystal the same way the scorer does it. Grey ribbon: the crystal protein."""
    ref_pdb = Path(row["ref_pdb"])
    ref_sdf = ref_pdb.with_name(ref_pdb.name.replace("_complex_ref.pdb", "_ligand_ref.sdf"))
    shown = [{"text": ref_pdb.read_text(), "fmt": "pdb",
              "styles": [({}, {"cartoon": {"color": "#d6d3ce", "opacity": 0.55}}),
                         ({"byres": True, "within": {"distance": 5, "sel": {"resn": "LIG"}}, "not": {"resn": "LIG"}},
                          {"line": {"color": "#9a958c"}})]},
             {"text": ref_sdf.read_text(), "fmt": "sdf", "styles": [({}, _stick(COLORS["crystal"]))]}]
    chips = [_chip(COLORS["crystal"], "crystal structure")]
    for k in models:
        cif = row.get(f"{k}_cif")
        if not isinstance(cif, str) or not Path(cif).exists():
            continue
        try:
            pdb = pose_in_crystal_frame(cif, ref_pdb)
        except Exception as e:
            chips.append(f"<i>{MODEL_NAMES.get(k, k)}: could not superpose ({e})</i>")
            continue
        shown.append({"text": pdb, "fmt": "pdb", "styles": [({"chain": "B"}, _stick(COLORS.get(k, "#2e9e4f")))]})
        rmsd = row.get(f"{k}_lig_rmsd")
        verdict = "" if pd.isna(rmsd) else f"  {rmsd:.1f} Å {'✓' if rmsd < 2 else '✗'}"
        chips.append(_chip(COLORS.get(k, "#2e9e4f"), MODEL_NAMES.get(k, k) + verdict))
    title = f"<b>{row['complex_name']}</b> · {row.get('stage', '')}"
    if pd.notna(row.get("experimental_pKD")):
        title += f" · measured affinity {row['experimental_pKD']:.1f}"
    # frame the crystal ligand and every predicted ligand, so a pose far from the pocket stays in view
    _viewer(shown, legend=title + "<br>" + "".join(chips), height=height, zoom="ligands")


def show_analog(cif_path, parent, label="new molecule (Boltz-2, live)", height=420):
    """A live prediction for a new molecule, moved onto the crystal of the compound it was
    made from: does the model keep the parent's binding mode?"""
    ref_pdb = Path(parent["ref_pdb"])
    ref_sdf = ref_pdb.with_name(ref_pdb.name.replace("_complex_ref.pdb", "_ligand_ref.sdf"))
    _viewer([{"text": ref_pdb.read_text(), "fmt": "pdb",
              "styles": [({}, {"cartoon": {"color": "#d6d3ce", "opacity": 0.55}})]},
             {"text": ref_sdf.read_text(), "fmt": "sdf", "styles": [({}, _stick(COLORS["crystal"]))]},
             {"text": pose_in_crystal_frame(str(cif_path), ref_pdb), "fmt": "pdb",
              "styles": [({"chain": "B"}, _stick("#2e9e4f"))]}],
            legend=_chip(COLORS["crystal"], f"parent {parent['complex_name']} (crystal)") + _chip("#2e9e4f", label),
            height=height, zoom="ligands")


def _cif_to_pdb(cif_text):
    """mmCIF text to PDB text (3Dmol draws ribbons more reliably from PDB)."""
    import io
    import tempfile
    from Bio.PDB import PDBIO
    from aiddd_pipeline import _load_cif
    with tempfile.NamedTemporaryFile("w", suffix=".cif") as f:
        f.write(cif_text)
        f.flush()
        model = _load_cif(f.name)
    for res in model.get_residues():
        if len(res.resname) > 3:
            res.resname = "LIG"
    out, buf = PDBIO(), io.StringIO()
    out.set_structure(model)
    out.save(buf)
    return buf.getvalue()


def show_prediction(cif_text, label="", height=380):
    """A live prediction on its own (no crystal to compare with): protein ribbon, ligand sticks."""
    _viewer([{"text": _cif_to_pdb(cif_text), "fmt": "pdb",
              "styles": [({}, {"cartoon": {"color": "#d6d3ce", "opacity": 0.7}}),
                         ({"chain": "B"}, _stick(COLORS["boltz2"]))]}],
            legend=_chip(COLORS["boltz2"], label or "Boltz-2 prediction"), height=height,
            zoom={"chain": "B"}, zoom_out=0.35)   # centred on the ligand, pocket in view


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def load_results(target):
    """Scored binding events from the full benchmark, on the curated set (valid crystal
    reference, no suspected crystallographic artefact), one row per crystal complex."""
    path = SHARED / "results" / target
    df = pd.concat([pd.read_parquet(f) for f in sorted(path.glob("*.parquet"))], ignore_index=True)
    meta = pd.read_csv(TARGETS[target].path("metadata"))
    keep = meta.pb_valid_ref.astype(bool) & ~meta.suspected_artefact.astype(bool)
    return df[df.complex_name.isin(set(meta.loc[keep, "complex_name"]))].reset_index(drop=True)


def success(rmsd):
    """Share of poses within 2 Å of the crystal, as a percentage."""
    rmsd = rmsd.astype(float).dropna()
    return round(100 * (rmsd < 2).mean(), 1) if len(rmsd) else float("nan")


def glance(results, models=("boltz2", "openbind0", "of3ft")):
    """% of poses under 2 Å per model, per target and campaign phase."""
    cols = {}
    for t, df in results.items():
        for phase, mask in (("fragments", df.fragment_screen), ("follow-ons", ~df.fragment_screen)):
            if mask.any():
                cols[f"{TARGETS[t].name}, {phase}"] = {MODEL_NAMES[m]: success(df.loc[mask, f"{m}_lig_rmsd"])
                                                       for m in models}
    return pd.DataFrame(cols)


def by_stage(df, models=("boltz2", "openbind0", "of3ft")):
    return pd.DataFrame({st: {MODEL_NAMES[m]: success(df.loc[df.stage == st, f"{m}_lig_rmsd"]) for m in models}
                         for st in STAGES if (df.stage == st).any()})


def serve_link(path="/"):
    """Where this Workspace exposes Ray Serve (port 8000) to your browser."""
    domain = os.environ.get("ANYSCALE_SESSION_DOMAIN")
    url = f"https://{domain}{path}" if domain else f"http://localhost:8000{path}"
    display(HTML(f'<p style="font-size:15px">Open the viewer: <a href="{url}" target="_blank"><b>{url}</b></a>'
                 f'<br><span style="color:#777">Or Workspace <b>Ports</b> tab → port 8000.</span></p>'))
    return url
