# Co-folding at scale: Boltz-2 and OpenBind-0 on Ray

**Can an AI model tell a chemist where their drug binds, accurately enough to design
the next molecule?** This workshop measures that on three real antiviral drug
discovery projects, 2,147 X-ray crystal structures in all, using every part of Ray:
Ray Data to predict and score, Ray Train to fine-tune, Ray Tune to tune, and Ray Serve
to serve live predictions.

New to the field? Every technical term is explained in
[Jargon in plain words](#jargon-in-plain-words).

---

## Why this matters

**Drugs are designed from poses.** A drug works by fitting into a pocket on a
protein. A chemist improving a drug candidate looks at *how* it sits in that pocket,
its **pose**: which atoms touch the protein, where there is empty space to grow into,
which group is pointing the wrong way. The next molecule is designed from that
picture. Get the pose wrong and the design is wrong: you grow the molecule into a wall.

**Crystal structures are the bottleneck.** The reliable way to see a pose is X-ray
crystallography. Each structure takes days to weeks, and many compounds never give a
usable crystal at all. A co-folding model (Boltz-2, OpenFold3) predicts the pose from
the protein sequence and the drug's formula in about a minute on one GPU. If the
prediction can be trusted, a project can look at every compound it is considering,
not just the few it managed to crystallize.

**Where in a project it matters.** A drug discovery project (a *campaign*) runs in two
phases:

| Phase | What happens | Do you need predicted poses? |
|---|---|---|
| **Fragment screen** | Hundreds of tiny molecules are soaked into protein crystals to find what sticks. Every hit gets a crystal structure as part of the experiment. | Not much: you already have the crystals. |
| **Follow-on design** | Chemists grow the best fragments into larger, tighter-binding molecules, hundreds of them over months. Only some are crystallized, late and at a cost. | **Yes.** This is where the money and time go, and where a reliable prediction saves the most. It is also the hard case: follow-ons contain chemistry no fragment showed. |

So the question this workshop asks is not "how good is the model on a benchmark?" but
**"how good is it on follow-on compounds, for my protein?"** Every result below is
split by campaign phase, and by protein.

**Why fine-tune.** General models learn from public structures, and your protein may
look nothing like what they saw. But every campaign starts with a fragment screen that
has already produced dozens of crystal structures *of your own protein*. Fine-tuning
spends that free data teaching a model your pocket, before the expensive follow-on
phase starts. The catch, which the results show: a model tuned to one protein can get
worse on others. It is a target-specific tool.

**Why tune.** Fine-tuning has settings, the most important being the learning rate:
too low and nothing changes, too high and the model forgets what it knew. Ray Tune
trains several versions side by side, stops the weak ones early, and picks the winner
by pose accuracy on held-out compounds. A second sweep asks whether a model that fails
on a protein can be rescued by giving it more compute at prediction time.

---

## Results at a glance

Share of predicted poses within 2 Å of the crystal (the usual cut-off for a correct
pose), on the curated set. Follow-on columns are the ones that matter most:

| Model | EV-A71 2A fragments | EV-A71 2A follow-ons | SARS-CoV-2 Mpro fragments | SARS-CoV-2 Mpro follow-ons | MERS-CoV Mpro follow-ons |
|---|---|---|---|---|---|
| Boltz-2 | 21.6% | 3.2% | 37.5% | 87% | 86.5% |
| OpenBind-0 | 28.4% | **92.7%** | 17.5% | **89%** | 83.3% |
| OF3 fine-tuned on EV-A71 2A fragments | 81.1%\* | 78.9% | 7.5% | 62% | 69.0% |

<sub>\* Trained on these structures, so not a held-out test. Every other cell is.</sub>

- **No model is best everywhere.** Boltz-2 is excellent on both coronavirus proteases
  and fails almost completely on EV-A71 2A follow-ons.
- **Fine-tuning works on its own target** (78.9% on EV-A71 2A follow-ons, against 3.2%
  for Boltz-2) **and costs accuracy elsewhere** (62 to 69% on the Mpro targets, against
  83 to 89% for the general models).
- **Lesson for a real project:** benchmark candidate models on your own protein's
  fragment screen before trusting any of them for follow-on design. This repository is
  that benchmark, and it runs on a new target by pointing it at new crystal data.

The three projects:

| Target | Source | Binding events | Compounds | With affinity |
|---|---|---|---|---|
| EV-A71 2A protease | OpenBind release | 925 | 699 | 601 (pKD) |
| SARS-CoV-2 main protease (Mpro) | ASAP Discovery and COVID Moonshot via Fragalysis | 628 | 308 | 164 (pIC50) |
| MERS-CoV main protease (Mpro) | ASAP Discovery via Fragalysis | 594 | 285 | 142 (pIC50) |

---

## The workshop in one animation

![The workshop end to end: question, crystal answer key, prediction, scoring, scale-up, workflow](docs/images/workshop.gif)

<sub>Real structures and real model outputs for EV-A71 2A compound x7259a. The
noise-to-pose motion in step 3 illustrates how diffusion models work; the final
poses are the models' actual predictions.</sub>

#### What the animation shows, step by step

The animation follows one real drug candidate, compound **x7259a**, through the whole
workshop. The bar along the bottom shows which of the six steps you are watching.

| Step | What you see | What it means |
|---|---|---|
| **1 Question** | A grey tangle of ribbon with a dashed circle. On the right, two lines of letters. | The ribbon is the **EV-A71 2A protease**, an enzyme the virus behind hand, foot and mouth disease needs to copy itself. Block it and the virus stalls. The dashed circle is the **binding pocket**, the groove where a drug can lodge. The two lines of letters are the *only* things a model is given: the protein's **amino-acid sequence** (its building blocks, one letter each) and the drug's **SMILES** (its chemical structure written as text). The label at the top says x7259a is a *late follow-on* with **pKD 7.9**: a compound from late in the project that binds very tightly. |
| **2 Answer key** | A black stick model appears inside the pocket. | This is where the drug really sits, measured by **X-ray crystallography**. It is the answer key. The models never see it. |
| **3 Predict** | Two coloured stick models start as scattered dots and settle into place: orange (OpenBind-0) and blue (Boltz-2). | Both models are **diffusion models**: they start from atoms placed at random and nudge them, step by step, into a 3D arrangement. The settling motion is an illustration of that idea; where each molecule ends up is the model's real prediction. Same inputs, same protein, but the two models put the drug in different places. |
| **4 Score** | Dotted grey lines connect each predicted atom to the matching crystal atom, and two numbers appear. | The predicted protein is first laid on top of the crystal protein. Then the error is measured as **ligand RMSD**: the typical distance between each predicted atom and where that atom really is. OpenBind-0 is off by 0.4 Å (Ångström, a tenth of a nanometre; a chemical bond is about 1.5 Å), which counts as a **correct pose**. Boltz-2 is off by 13.2 Å: it put the drug in a different part of the protein altogether. Anything under **2 Å** counts as correct. |
| **5 Scale up** | Grids of small squares fill in, one row per model and one block per protein. | Every square is one crystal structure, in the order the project made them: the **fragment screen** first, then **early** and **late follow-ons**. A filled square means that model got the pose right (under 2 Å), a pale square means it did not, and the percentage is the share it got right. Ray Data runs all three models over every structure at once on a cluster of GPUs. Read across the rows: Boltz-2 misses almost every EV-A71 2A compound but gets most SARS-CoV-2 ones right, while the model fine-tuned on EV-A71 2A does the opposite. No model is best everywhere. |
| **6 Workflow** | Five boxes joined by arrows. | What you build in the workshop: download public **data**, **predict** with Ray Data, **score** with Ray Data, **fine-tune** a model with Ray Train (and tune it with Ray Tune), and **serve** live predictions and a 3D viewer with Ray Serve. |

**Fragments and follow-ons, in one real series.** A fragment from the EV-A71 2A
screen grown into a potent follow-on compound, and how each model does at each stage:

![A fragment grown into a potent follow-on compound, and how each model does at each stage](docs/images/campaign.gif)

---

## Run everything

Prerequisite: an Anyscale Workspace on image `anyscale/image/aiddd-boltz2-of3:4` with
this repository in it ([Step 1](#step-1-workspace-and-image)). Run every command from
the repository root in the Workspace terminal.

Everything except the viewer is an **Anyscale Job**: `anyscale job submit -f job.yaml`
uploads this folder, starts its own autoscaling cluster (L4 GPU and CPU nodes, all
from zero), runs the command after `--`, and shuts the cluster down when it finishes.
Your Workspace can be closed while jobs run. Follow a job in the console link the
command prints, or with `anyscale job logs --id <job id> --follow`. All data and
results go to shared storage, `/mnt/user_storage/aiddd`, which the Workspace and
every job can read.

| # | What | Command | Time |
|---|---|---|---|
| 2 | Download data and weights (once) | `anyscale job submit -f job.yaml --name aiddd-setup -- python setup_data.py` | 15 to 20 min |
| 2b | Build the two Mpro targets (once) | `anyscale job submit -f job.yaml --name aiddd-mpro-setup -- python prepare_mpro.py` | 10 min |
| 3 | Smoke test: 8 ligands, 3 models | `anyscale job submit -f job.yaml --name aiddd-smoke -- python aiddd_pipeline.py --models boltz2,openbind0,of3ft --limit 8 --max-gpus 2 --out /mnt/user_storage/aiddd/results/smoke` | 15 to 20 min |
| 4 | **Benchmark** EV-A71 2A (Ray Data) | `anyscale job submit -f job.yaml --name aiddd-ev71` | 1 to 2 h on 8 L4 |
| 4 | **Benchmark** SARS-CoV-2 Mpro | `anyscale job submit -f job.yaml --name aiddd-sars2 -- python aiddd_pipeline.py --target sars2-mpro --models boltz2,openbind0,of3ft` | 1 to 2 h on 8 L4 |
| 4 | **Benchmark** MERS-CoV Mpro | `anyscale job submit -f job.yaml --name aiddd-mers -- python aiddd_pipeline.py --target mers-mpro --models boltz2,openbind0,of3ft` | 1 to 2 h on 8 L4 |
| 5 | Compare all targets | `anyscale job submit -f job.yaml --name aiddd-compare -- python compare_targets.py --csv /mnt/user_storage/aiddd/results/summary.csv` | 2 min |
| 6 | **Viewer and live prediction** (Ray Serve), in the Workspace | `python check_serve.py`, then `serve run serve_viewer:app` and open port 8000 | 5 min |
| 7 | Build the fine-tuning set | `anyscale job submit -f job.yaml --name aiddd-ft-data -- python prepare_finetune.py --check` | 5 min |
| 7 | **Fine-tune** OpenFold3 (Ray Train) | `anyscale job submit -f job.yaml --name aiddd-finetune -- python finetune_ray_train.py --num-workers 2 --max-steps 150` | 30 min on 2 L4 |
| 8 | **Tune** the fine-tune (Ray Tune + Train) | `anyscale job submit -f job.yaml --name aiddd-tune-ft -- python tune_finetune.py` | about 45 min on 4 L4 |
| 8 | **Tune** Boltz-2 settings (Ray Tune) | `anyscale job submit -f job.yaml --name aiddd-tune-boltz -- python tune_boltz.py` | about 45 min on 8 L4 |

The three benchmark jobs are independent and can run at the same time, as can the
two Tune sweeps. Steps 7 and 8 need only step 2. Every job is resumable: predictions
are cached per ligand, so a resubmitted job only predicts what is missing.

The sections below explain each step, what success looks like, and how to read the
output.

---

## Contents

- [Runbook](#runbook)
  - [Step 1. Workspace and image](#step-1-workspace-and-image)
  - [Step 2. Download data and weights](#step-2-download-data-and-weights)
  - [Step 2b. Build the Mpro targets](#step-2b-build-the-mpro-targets)
  - [Step 3. Smoke test](#step-3-smoke-test)
  - [Step 4. Full benchmark](#step-4-full-benchmark)
  - [Step 5. Read the results](#step-5-read-the-results)
  - [Step 6. Viewer and live prediction](#step-6-viewer-and-live-prediction)
  - [Step 7. Fine-tune with Ray Train](#step-7-fine-tune-with-ray-train)
  - [Step 8. Tune with Ray Tune](#step-8-tune-with-ray-tune)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [How it works](#how-it-works)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Jargon in plain words](#jargon-in-plain-words)
- [Data sources and licenses](#data-sources-and-licenses)

---

## Runbook

### Step 1. Workspace and image

Create an Anyscale Workspace with:

| Setting | Value |
|---|---|
| Container image | `anyscale/image/aiddd-boltz2-of3:4` |
| Head node | `m5.2xlarge` (CPU only, so notebooks never hold a GPU) |
| Worker group | `g6.xlarge` or `g6.2xlarge` (1x L4), min 0, max 4. Needed for live prediction in step 6 |

Clone or upload this repository into the Workspace.

To build the image yourself instead:

```bash
anyscale image build -f image/Containerfile -n aiddd-boltz2-of3 -r 2.59.0
```

Then update `image_uri` in `job.yaml` to the version it prints.

> **Why Python 3.12:** Boltz-2 pins an older SciPy that does not build on 3.13.
> The image also carries a second, isolated OpenFold3 0.4.3 install at
> `/opt/of3-legacy`, because the released fine-tuned checkpoint only loads there.

### Step 2. Download data and weights

All data lives in `/mnt/user_storage/aiddd`, which both Workspaces and Jobs can read.

Run it as a Job, so the download does not depend on your Workspace session:

```bash
anyscale job submit -f job.yaml --name aiddd-setup -- python setup_data.py
```

Or directly in the Workspace terminal:

```bash
python setup_data.py
```

What you get:

```
/mnt/user_storage/aiddd/
  raw/OpenBind_EV-A71_2A/      structures, metadata CSV, FASTA, MSA        (500 MB)
  EV-A71_2A_benchmark/         published OpenBind results, for comparison  (11 MB)
  boltz_cache/                 Boltz-2 weights and chemical component data (6 GB)
  of3_cache/                   OpenBind-0 and fine-tuned OF3 checkpoints   (9 GB)
```

The script skips anything already present, so it is safe to rerun. Use
`--skip-of3` if you only want Boltz-2.

### Step 2b. Build the Mpro targets

```bash
anyscale job submit -f job.yaml --name aiddd-mpro-setup -- python prepare_mpro.py
```

This downloads the ASAP Discovery and COVID Moonshot crystal structures from
Fragalysis, joins the ASAP potency data, builds one MSA per protein, and writes
each target in the same layout as the OpenBind release:

```
/mnt/user_storage/aiddd/raw/SARS-CoV-2_Mpro/    metadata.csv, protein.fasta, protein.a3m, structures/
/mnt/user_storage/aiddd/raw/MERS-CoV_Mpro/
```

How campaign stage is defined for Mpro:

| Stage | SARS-CoV-2 Mpro | MERS-CoV Mpro |
|---|---|---|
| fragment | COVID Moonshot XChem fragment screen (16 heavy atoms or fewer) | none |
| early follow-on | ASAP compounds in the Polaris training set | same |
| late follow-on | ASAP compounds in the Polaris test set, made later in the campaign | same |

ASAP compounds without potency data are staged by crystal order. Binding events
more than 8 A from the catalytic cysteine (crystal contacts, surface sites) are
excluded from the curated benchmark.

### Step 3. Smoke test

Eight ligands (two fragments, six follow-ons) through all three models:

```bash
anyscale job submit -f job.yaml --name aiddd-smoke -- \
  python aiddd_pipeline.py --models boltz2,openbind0,of3ft \
    --limit 8 --max-gpus 2 --out /mnt/user_storage/aiddd/results/smoke
```

**What success looks like** at the end of the job log:

```
Done in N min. 10 complexes scored -> /mnt/user_storage/aiddd/results/smoke
  Boltz-2: {'ok': 8}
  OpenBind-0 (OF3): {'ok': 8}
  OF3 fine-tuned on EV-A71 2A: {'ok': 8}
```

`cached` instead of `ok` is also a success: that ligand was predicted by an
earlier run. Any `failed` count prints the first error underneath; see
[Troubleshooting](#troubleshooting).

### Step 4. Full benchmark

One job per target:

```bash
anyscale job submit -f job.yaml --name aiddd-ev71        # EV-A71 2A (the job.yaml default)
anyscale job submit -f job.yaml --name aiddd-sars2 -- \
  python aiddd_pipeline.py --target sars2-mpro --models boltz2,openbind0,of3ft --max-gpus 8
anyscale job submit -f job.yaml --name aiddd-mers -- \
  python aiddd_pipeline.py --target mers-mpro --models boltz2,openbind0,of3ft --max-gpus 8
```

Each job runs on its own cluster, so the three can run at the same time.
Results go to `/mnt/user_storage/aiddd/results/<target>`.

Open the Ray Dashboard from the Job page while it runs. You will see L4 and CPU
nodes join as the queue fills and leave when their stage drains.

The job is resumable. Predictions are cached per ligand under
shared storage (`/mnt/user_storage/aiddd/predictions/`), so resubmitting after an
interruption only predicts what is missing.

### Step 5. Read the results

The job log ends with a summary on the curated benchmark set (PoseBusters valid
reference, no suspected crystallographic artefact):

```
Boltz-2      fragments                n= ...  median RMSD ... A  <2A ...%   | published (same complexes) ...
Boltz-2      follow-ons               n= ...
Boltz-2      stage: fragment          ...
Boltz-2      stage: early follow-on   ...
Boltz-2      stage: late follow-on    ...
Boltz-2      sim: far (<0.3)          ...
...
Boltz-2 affinity vs experimental pKD: Spearman ... (n=... ligands)
```

![Pose accuracy by target and campaign stage](docs/images/pose_accuracy.png)

**Reading the figure.** Panels a and b show, for every error threshold on the x axis,
the share of predicted poses within that error of the crystal (higher and further
left is better). The dashed line is the usual 2 Å cutoff for a correct pose. Panels c
and d split the 2 Å success rate by campaign stage. Note that the fine-tuned model was
trained on the EV-A71 2A fragment screen, so its EV-A71 2A fragment score (panel c) is
not a held-out test; its follow-on, SARS-CoV-2 and MERS-CoV scores are.

**Reference results** from our runs are in [Results at a glance](#results-at-a-glance); your numbers should be close, since each model samples one pose per ligand.

On EV-A71 2A follow-ons the published numbers for the same complexes are 7.1%
(Boltz-2) and 63.4% (fine-tuned OF3). Boltz-2's predicted affinity ranks compounds
with Spearman 0.41 against pKD on EV-A71 2A (482 ligands), 0.86 against pIC50 on
SARS-CoV-2 Mpro (164 ligands) and 0.59 against pIC50 on MERS-CoV Mpro (142 ligands).
MERS-CoV Mpro has no fragment screen, so it has follow-on results only; it behaves like
its close relative SARS-CoV-2 Mpro, and the model fine-tuned on EV-A71 2A again trails
on a protease it never saw.

![Boltz-2 predicted affinity against measured affinity](docs/images/affinity.png)

A Spearman correlation of 1 would mean the model ranks compounds exactly in the
measured order; 0 means no relation. On SARS-CoV-2 Mpro the predictions fall into two
clusters (weaker and stronger binders), and part of the 0.86 comes from separating
them: within the lower and upper cluster the correlation is 0.65 and 0.38.

How to read it:

| Term | Meaning |
|---|---|
| Ligand RMSD | Heavy atom distance between predicted and crystal ligand, after aligning the protein on its CA atoms |
| < 2 A | The standard cutoff for a correct pose |
| published | The OpenBind authors' own number for the same method on the same complexes. Your run should land close to it |
| stage | Fragment screen, then follow-on compounds split at the median crystal id. Crystal ids follow campaign time |
| sim | Highest fingerprint similarity (Morgan, Tanimoto) to any fragment-screen ligand. Low similarity means chemistry the fragment screen never showed |

For all targets in one table:

```bash
anyscale job submit -f job.yaml --name aiddd-compare -- \
  python compare_targets.py --csv /mnt/user_storage/aiddd/results/summary.csv
```

The questions this answers:

- **Generalization:** does pose accuracy drop for follow-on compounds, and for compounds far from the fragments?
- **Across targets:** is a model that does well on one protease equally good on the others?
- **Target adaptation:** does the fine-tuned checkpoint (trained only on EV-A71 2A fragment structures) close that gap, and does it hurt the Mpro targets it never saw?
- **Affinity:** does Boltz-2's predicted affinity rank compounds the way the experiments do?

The full per-complex table is Parquet under `--out`:

```python
import pandas as pd
df = pd.read_parquet("/mnt/user_storage/aiddd/results/ev71-2a")
df.groupby("stage")[["boltz2_lig_rmsd", "openbind0_lig_rmsd", "of3ft_lig_rmsd"]].median()
```

### Step 6. Viewer and live prediction

First confirm everything works end to end (deploys the app, loads the
leaderboard, runs one live prediction, repeats it from cache, rejects a bad SMILES):

```bash
python check_serve.py
```

Then start the app from the Workspace terminal:

```bash
AIDDD_RESULTS=/mnt/user_storage/aiddd/results/ev71-2a serve run serve_viewer:app
```

Open port 8000 from the Workspace **Ports** tab. Point `AIDDD_RESULTS` at
`results/sars2-mpro` or `results/mers-mpro` to browse the other targets.

- **Leaderboard:** every crystal complex, sorted by ligand RMSD, with model tabs at the top.
- **3D viewer:** click a row to see the predicted ligand (green) over the crystal ligand (orange).
- **Live prediction:** paste a SMILES and press Predict. Boltz-2 runs on an L4 replica
  and returns the pose, ligand ipTM, predicted affinity, and binder probability.
  A new molecule takes about a minute; repeats come back instantly from cache.

The live deployment autoscales from one to four L4 replicas with traffic.

### Step 7. Fine-tune with Ray Train

The `of3ft` stage evaluates a released checkpoint that was fine-tuned on EV-A71 2A
fragment structures. This step shows how such a checkpoint is made: OpenFold3 is
fine-tuned from the OpenBind-0 weights on the fragment screen, with Ray Train
running OpenFold3's own PyTorch Lightning training loop on several GPUs.

**1. Build the training set** (CPU, once):

```bash
anyscale job submit -f job.yaml --name aiddd-ft-data -- python prepare_finetune.py --check
```

This selects the fragment-screen complexes with a valid reference pose, no suspected
artefact and no covalent ligand, one per compound (66 complexes), and writes them in
the format OpenFold3's training data loader reads, under
`/mnt/user_storage/aiddd/finetune/ev71-2a/`. `--check` loads a few samples back
through OpenFold3's own `WeightedPDBDataset` to prove the format.

**2. Train** (2 L4 GPUs, about 30 min including node startup):

```bash
anyscale job submit -f job.yaml --name aiddd-finetune -- \
  python finetune_ray_train.py --num-workers 2 --max-steps 150
```

The log shows one line per optimizer step, averaged over the workers, then a summary.
From a reference run (2x L4, 150 steps):

```
step  150  loss   0.132  avg10   0.113  distogram  1.711  lr 1.0e-04    8.2 s  13.6 GB

Done in 27.7 min: 150 steps on 2 workers
  loss                           first 10 steps 0.138 -> last 10 steps 0.113
  distogram_loss                 first 10 steps 1.424 -> last 10 steps 1.166
  plddt_loss                     first 10 steps 1.762 -> last 10 steps 1.317
  lddt_inter_protein_ligand_step first 10 steps 0.434 -> last 10 steps 0.576
  median step time 9.2 s, peak GPU memory 13.6 GB
  weights: /mnt/user_storage/aiddd/finetune/runs/<name>/checkpoint_.../of3_ev71_ft.pt
```

How it works:

| Piece | Role |
|---|---|
| `TorchTrainer` + `ScalingConfig(num_workers=N, use_gpu=True)` | One training worker per L4, NCCL process group, nodes scale up from zero |
| `RayDDPStrategy`, `RayLightningEnvironment` | Run OpenFold3's unchanged Lightning training loop under Ray instead of torchrun or SLURM |
| `ray.train.report` | Loss terms, learning rate, step time and GPU memory after every step (visible in the Ray Dashboard Train view) |
| Final checkpoint | Reported to Ray once at the end. A plain OpenFold3 state dict, the same format as OpenBind-0 |

Training settings live in `finetune/finetune_ev71.yml`, a standard OpenFold3 runner
YAML. The defaults fit one 24 GB L4 per worker: a 192-token budget holds a whole
complex without cropping, and 8 diffusion noise levels per step instead of 48.

| Flag | Default | Description |
|---|---|---|
| `--num-workers` | 2 | GPUs, one complex per GPU per step |
| `--max-steps` | 150 | Optimizer steps |
| `--lr` | 1e-4 | Peak learning rate, reached after `--warmup-steps` (10) |
| `--diffusion-samples` | 8 | Noise levels per complex per step |
| `--export-ema` | off | Export the EMA weights instead of the raw weights |

![Ray Train fine-tuning curves](docs/images/finetune.png)

**Reading the curve.** OpenBind-0 already predicts these fragment complexes well, so
the starting loss is low, and single steps are noisy: each worker sees a different
complex and the diffusion loss depends on the noise level drawn at random each step.
Leaving out the first step (it includes kernel compilation), the training loss is
nearly flat over 150 steps (0.120 to 0.117, first vs last 20 steps). The clearer
signal is `lddt_inter_protein_ligand_step`, the protein-ligand lDDT of a short
structure rollout during training: it rises from 0.44 to 0.57. This is a
demonstration scale run that shows the Ray Train mechanics; the released `of3ft`
checkpoint was trained far longer (1,000 steps on 8 GPUs).

**3. Benchmark your weights.** The pipeline has a fourth model, `of3ray`, that uses
the most recent weights from this step (or `--of3ray-ckpt <file>`):

```bash
anyscale job submit -f job.yaml --name aiddd-of3ray -- \
  python aiddd_pipeline.py --models of3ray,of3ft,openbind0
```

This puts your fine-tune next to its starting point (OpenBind-0) and the released
fine-tune (`of3ft`) in the same summary table. Predictions are cached per ligand,
so delete `/mnt/user_storage/aiddd/predictions/of3ray` after retraining.

### Step 8. Tune with Ray Tune

Two sweeps, each answering a question the earlier steps leave open. Both run as
Anyscale Jobs and can run at the same time.

#### 8a. Which learning rate gives the best fine-tune? (Ray Tune + Ray Train)

Step 7 used one learning rate. Too low and the model barely changes; too high and it
forgets what OpenBind-0 already knew. Training loss cannot tell you which is right
(it is nearly flat, see step 7), so this sweep judges each run by what we actually
care about: **pose accuracy on follow-on compounds it never trained on**.

```bash
anyscale job submit -f job.yaml --name aiddd-tune-ft -- python tune_finetune.py
```

What happens:

1. **One Tune trial per learning rate** (default `3e-5, 1e-4, 3e-4, 1e-3`). Each trial
   is a complete Ray Train fine-tune from step 7, on its own L4. Up to four run at once.
2. **ASHA early stopping.** Every training step is streamed to Tune
   (`TuneReportCallback`). After a grace period of 60 steps, Tune compares the trials'
   smoothed protein-ligand lDDT and stops the ones falling behind, freeing their GPUs.
3. **Validation on held-out poses.** Each trial that finishes predicts 24 EV-A71 2A
   *early* follow-on compounds with its new weights, on a fresh GPU, and scores them
   against the crystals.
4. **Pick the winner** by validation pose accuracy. The *late* follow-ons are never
   touched, so benchmarking the winner on them afterwards is still a fair test.

The log ends with the starting points and one line per trial:

```
Validation set: 24 early follow-on ligands, ... crystal complexes (EV-A71 2A). Late follow-ons stay held out.
  OpenBind-0 (starting weights)     ...% under 2 A, median ... A
  released fine-tune (of3ft)        ...% under 2 A, median ... A

  lr 3e-05  150 steps  lDDT ...  validation ...% under 2 A, median ... A
  lr 1e-03  stopped early by ASHA at step 60 (lDDT ...)
  ...
Best: lr ...
  weights: /mnt/user_storage/aiddd/finetune/runs/tune-of3-ev71-.../trial_.../checkpoint_.../of3_ev71_ft.pt
  benchmark it: python aiddd_pipeline.py --models of3ray,openbind0 --of3ray-ckpt <weights>
```

Benchmark the winner on every EV-A71 2A compound with the command it prints, as a job:

```bash
anyscale job submit -f job.yaml --name aiddd-of3ray-best -- \
  python aiddd_pipeline.py --models of3ray,openbind0 --of3ray-ckpt <weights> \
    --out /mnt/user_storage/aiddd/results/ev71-2a-tuned
```

| Piece | Role |
|---|---|
| `tune.Tuner(trial, param_space={"lr": tune.grid_search([...])})` | One trial per learning rate, run concurrently |
| `TorchTrainer(..., RunConfig(callbacks=[TuneReportCallback()]))` inside each trial | The step 7 training run, reporting every step to Tune |
| `ASHAScheduler(metric="lddt_avg20", grace_period=60)` | Stops trials that fall behind, so GPUs go to the promising ones |
| Ray task with `num_gpus=1` after training | Predicts and scores the validation compounds with the trial's weights |

| Flag | Default | Description |
|---|---|---|
| `--lrs` | `3e-5,1e-4,3e-4,1e-3` | Learning rates, one trial each |
| `--max-steps` | 150 | Training steps per trial |
| `--workers-per-trial` | 1 | GPUs per trial |
| `--max-concurrent` | 4 | Trials running at the same time |
| `--grace-steps` | 60 | Steps every trial runs before ASHA may stop it |
| `--no-early-stop` | off | Run every trial to the end |
| `--val-ligands` | 24 | Held-out early follow-on ligands used to pick the winner |

#### 8b. Can more compute rescue Boltz-2? (Ray Tune)

Boltz-2 gets only 3.2% of EV-A71 2A follow-on poses right. Is that because it did not
try hard enough (one sample, three recycling steps), or because it lacks physics, or
neither? This sweep runs Boltz-2 with eight combinations of settings on the same 32
follow-on compounds, one L4 per setting, all at once:

```bash
anyscale job submit -f job.yaml --name aiddd-tune-boltz -- python tune_boltz.py
```

| Setting | Values tried | What it does |
|---|---|---|
| Recycling steps | 3 (default), 10 | Passes the structure back through the model to refine it |
| Diffusion samples | 1 (default), 5 | Draws several poses and keeps the one Boltz-2 is most confident in |
| Steering (`--use_potentials`) | off (default), on | Pushes atoms away from physically impossible arrangements while the pose forms |

Each trial reports its running accuracy to Tune after every batch of 8 ligands, so
you can watch the sweep in the Ray Dashboard. The log ends with one line per setting,
best first:

```
Done in ... min. Boltz-2 on ev71-2a follow-ons, share of poses under 2 A:
  rec10_samples5_steered        ...%  median ... A  n=...  ... min
  rec3_samples1                 ...%  median ... A  n=...  ... min
  ...
```

Use `--target sars2-mpro` or `mers-mpro` to run the same sweep where Boltz-2 already
does well, and `--recycling`, `--samples`, `--steering` to change the grid.
Predictions are cached per setting under `/mnt/user_storage/aiddd/tune/boltz2/`.

---

## Architecture

One Ray Data pipeline per model, all running at the same time on one autoscaling
cluster. Each model gets a fixed share of the GPUs, so no model waits for another.

![Cluster architecture](docs/images/architecture.png)

The scored results feed a Ray Serve app with a leaderboard, a 3D pose overlay, and
live Boltz-2 prediction on an L4.

## Repository layout

| File | Purpose |
|---|---|
| `setup_data.py` | One-time download of the EV-A71 dataset, published benchmark, and model weights |
| `prepare_mpro.py` | One-time build of the SARS-CoV-2 and MERS-CoV Mpro targets |
| `aiddd_pipeline.py` | The Ray Data benchmark pipeline (all stages above), one target per run |
| `compare_targets.py` | One table across targets, models, and campaign stages |
| `serve_viewer.py` | Ray Serve app: results browser, 3D viewer, live prediction |
| `check_serve.py` | End to end check of the Serve app, including one live prediction |
| `prepare_finetune.py` | Builds the OpenFold3 training set from the EV-A71 2A fragment screen |
| `finetune_ray_train.py` | Fine-tunes OpenFold3 with Ray Train (TorchTrainer, Lightning DDP) |
| `finetune/finetune_ev71.yml` | OpenFold3 training settings used by the fine-tune |
| `tune_finetune.py` | Ray Tune sweep over the fine-tune's learning rate (Tune + Train, ASHA early stopping, pose validation) |
| `tune_boltz.py` | Ray Tune sweep over Boltz-2 inference settings (recycling, samples, steering) |
| `job.yaml` | Anyscale Job definition with autoscaling L4 and CPU worker groups |
| `image/Containerfile` | Container image: Ray 2.59, Python 3.12, Boltz-2, OpenFold3 |
| `docs/images/` | Figures in this README, drawn from the runs described here |
| `docs/figures/` | Scripts that redraw every figure and GIF from the results under `/mnt/user_storage/aiddd` |

---

## How it works

**One prediction per ligand, scored against every crystal.** On EV-A71 2A, 925
binding events come from 699 unique ligands (some crystals hold two copies). The
pipeline predicts once per ligand and scores against each binding event, which
saves about 25% of GPU time.

**Batching amortizes model loading.** Each GPU actor receives a batch of ligands and
runs one `boltz predict` or `run_openfold predict` call for the whole batch, so model
weights load once per batch instead of once per ligand.

**Several predictors per GPU.** One complex of 150 to 300 residues uses under 5 GB
and cannot keep an L4 busy alone: the GPU idles while a predictor loads weights and
featurizes on CPU, and OpenFold3's small kernels leave it at about 35% even while
running. The pipeline therefore runs 3 OpenFold3 or 2 Boltz-2 predictors per GPU
(fractional `num_gpus` in Ray Data), so one computes while another prepares.
Measured on one L4, EV-A71 2A, 8 ligands per batch:

| Predictors per L4 | GPU activity | Seconds per ligand | GPU memory |
|---|---|---|---|
| OpenFold3, 1 | 32% | 19.9 | 2.9 GB |
| OpenFold3, 2 | 76% | 11.0 | 7.8 GB |
| OpenFold3, 3 | 82% | 9.5 | 10.1 GB |
| Boltz-2, 1 | 75% | 40.8 | 4.8 GB |
| Boltz-2, 2 | 83% | 43.3 | 9.1 GB |

OpenFold3 gets twice the throughput. Boltz-2 does not: more than half its time is
the affinity step, which already keeps the GPU 98% busy. Packing still fills its
idle startup gaps, so the default is 2.

![Throughput on one L4 when several predictors share the GPU](docs/images/throughput.png)

**One pipeline per model, run concurrently.** Each model reads the validated ligand
list directly and runs as its own Ray Data pipeline. The driver starts them together
and joins the results before scoring. Chaining the models as stages of a single
pipeline makes every stage run at the speed of the slowest one: in a chained test
run, four Boltz-2 GPUs sat at 0% for over ten minutes waiting on an OpenFold3 stage.

**Fixed GPU shares.** `--max-gpus` is the GPU budget for the whole run. It is split
across models by measured cost per ligand (Boltz-2 costs about twice an OpenFold3
model), and each model gets a fixed actor pool of that size
(`ActorPoolStrategy(size=n, max_tasks_in_flight_per_actor=1)`). Ray Data adds nodes
when CPU or memory is busy, so fractional GPU actors that fit on the first nodes
would never trigger a scale up; a fixed pool is always requested, so the cluster
grows straight to the budget. One batch in flight per actor stops batches queueing
behind a busy actor while another sits idle at the end of the run.

**GPU utilization report.** The first predictor on each GPU node samples
`nvidia-smi` every 5 s, and the job log ends with the mean utilization per node.
Measured on MERS-CoV Mpro with 8 L4s, once all nodes were up:

| Pipeline design | GPU utilization |
|---|---|
| One predictor per GPU, models chained | about 60% across the cluster |
| Packed, models chained | 0% on the 4 Boltz-2 nodes, 43 to 92% on the others |
| Packed, one pipeline per model (default) | 93% on average while all eight GPUs were up |

![GPU utilization per node across the run](docs/images/gpu_cluster.png)

**Boltz-2 affinity ensemble.** `--affinity-samples` sets how many diffusion samples
the Boltz-2 affinity head uses (default 5). On 160 EV-A71 2A ligands with measured
pKD, one sample ranks compounds as well as five (Spearman 0.37 vs 0.36 against
experiment, and the two agree at Pearson 0.99).

**Right hardware per stage.** Boltz-2 and OpenFold3 inference both fit on a 24 GB
L4 for these proteins (150 to 306 residues). For larger proteins, add an L40S
worker group to `job.yaml` and pass `--of3-accelerator L40S`. Validation and
scoring run on CPU nodes; Ray Data places each model's actors with
`accelerator_type`.

**One MSA per target.** Every complex of a target shares one protein, so a single
precomputed MSA is reused for every prediction. No MSA server is called at runtime.

**Pose scoring.** The predicted protein is aligned on CA atoms to the crystal chain
the ligand actually contacts (many crystals contain two protein copies, and aligning
to the wrong one adds 30 to 40 A). Ligand RMSD uses element-matched optimal
assignment, which handles symmetric groups without needing bond orders.

**Caching and resume.** Each ligand's structure (CIF) and metrics (JSON) are written
to shared storage as soon as they finish. Reruns skip them.

---

## Configuration reference

### `aiddd_pipeline.py`

| Flag | Default | Description |
|---|---|---|
| `--target` | `ev71-2a` | `ev71-2a`, `sars2-mpro`, or `mers-mpro` |
| `--models` | `boltz2,openbind0` | Any of `boltz2`, `openbind0`, `of3ft`, `of3ray` |
| `--limit` | all | First N ligands, a fixed mix of fragments and follow-ons |
| `--batch-size` | 8 | Ligands per GPU call. 8 to 16 works well |
| `--boltz-per-gpu` | 2 | Boltz-2 predictors sharing one GPU |
| `--of3-per-gpu` | 3 | OpenFold3 predictors sharing one GPU |
| `--max-gpus` | 8 | GPUs for the whole pipeline, split across model stages by cost |
| `--accelerator` | `L4` | GPU type for Boltz-2 (and OF3 unless overridden) |
| `--of3-accelerator` | same as `--accelerator` | GPU type for the OpenFold3 stages, for example `L40S` |
| `--diffusion-samples` | 1 | Poses sampled per ligand. More samples cost proportionally more GPU time |
| `--affinity-samples` | 5 | Boltz-2 affinity ensemble size. Lower is faster; results are cached separately |
| `--affinity-only` | off | Only ligands with an experimental affinity (with `--limit`, a random sample) |
| `--of3ray-ckpt` | latest Ray Train run | Weights for the `of3ray` model |
| `--out` | `$AIDDD_SHARED/results/<target>` | Output Parquet directory |

### Environment variables

| Variable | Default | Used by |
|---|---|---|
| `AIDDD_SHARED` | `/mnt/user_storage/aiddd` | All scripts: root of data, weights, predictions, results |
| `AIDDD_RESULTS` | `$AIDDD_SHARED/results/ev71-2a` | `serve_viewer.py`: which results to show |

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Missing /mnt/user_storage/aiddd/...` | Data not downloaded yet. Run step 2 (EV-A71) or step 2b (Mpro) |
| Job sits at `pending=1` actors for several minutes | GPU nodes are starting. First node startup takes 3 to 6 minutes |
| `InsufficientInstanceCapacity` in the log | The cloud is out of that GPU type. `enable_cross_zone_scaling` retries other zones. If every zone fails (L40S is often scarce), drop `--of3-accelerator L40S` so everything runs on L4 |
| `Default checkpoint ... not found` from OpenFold3 | `of3_cache` is incomplete. Rerun `python setup_data.py` |
| `of3ft` fails with missing or unexpected state dict keys | The job is not using the project image, so the 0.4.3 install at `/opt/of3-legacy` is missing. Check `image_uri` |
| CUDA out of memory in an OpenFold3 stage | Lower `--batch-size`, or add an L40S group and pass `--of3-accelerator L40S` |
| Viewer shows `not scored yet` | `AIDDD_RESULTS` points to a run without that model. Pick another model tab or results directory |
| Live prediction never returns | No L4 available to the Workspace. Add an L4 worker group (step 1) |

---

## Jargon in plain words

### Biology and chemistry

| Term | Plain meaning |
|---|---|
| **Protein** | A large molecule made by living things (and viruses) that does a job: cutting, copying, signalling. Built from a chain of amino acids that folds into a specific 3D shape. |
| **Amino acid**, **sequence** | The 20 building blocks of proteins. A protein's *sequence* is the order of its amino acids, written one letter each (`GKFGQQSG...`). |
| **Protease** | A protein whose job is to cut other proteins. Many viruses make one long protein and need their own protease to cut it into working pieces, so blocking the protease stops the virus. All three targets here are viral proteases. |
| **Target** | The protein a drug is meant to act on. This workshop has three: EV-A71 2A protease, SARS-CoV-2 Mpro and MERS-CoV Mpro. |
| **EV-A71** | Enterovirus A71, a cause of hand, foot and mouth disease. **2A** is the name of one of its proteases. |
| **SARS-CoV-2**, **MERS-CoV** | The coronaviruses behind COVID-19 and MERS (Middle East respiratory syndrome). |
| **Mpro** | *Main protease*: the coronavirus protease that does most of the cutting. The COVID drug Paxlovid blocks it. |
| **Ligand** | Any small molecule that binds to a protein. In this workshop, ligand simply means the drug candidate. |
| **Compound** | One specific molecule a chemist made or bought. "Compound x7259a" is one ligand. |
| **Binding pocket**, **binding site** | The groove on the protein surface where the ligand sits. |
| **Complex**, **protein-ligand complex** | The protein with the ligand bound to it, as one structure. |
| **Binding event** | One ligand copy seen in one crystal. A crystal can hold two copies of the protein, so one compound can give two binding events. |
| **Pose** | Where the ligand sits in the pocket and which way it faces. The thing the models predict. |
| **SMILES** | A way to write a molecule's structure as a line of text, e.g. `CC(=O)O` for acetic acid. It is the model's only input about the drug. |
| **Heavy atom** | Any atom except hydrogen. Hydrogens are too light to see in most crystal structures, so they are left out of scoring. |
| **Covalent ligand** | A ligand that forms a permanent chemical bond with the protein instead of just sitting in the pocket. |
| **Catalytic cysteine** | The amino acid at the heart of the Mpro pocket that does the cutting. Ligands far from it are not in the drug pocket. |
| **X-ray crystallography**, **crystal structure** | The lab method that gives the true 3D positions of every atom: grow a crystal of the protein, soak in the ligand, shine X-rays through it and work back to the structure. Slow and expensive, but the ground truth here. |
| **Fragalysis** | A public website from the Diamond Light Source synchrotron that hosts fragment-screen crystal structures, including the Mpro ones used here. |
| **MSA** (multiple sequence alignment) | The protein's sequence lined up against related proteins from other organisms. Which positions change together hints at which parts touch in 3D, so co-folding models use it as extra input. Files: `.a3m`. |
| **FASTA** | A plain text file holding a sequence. |

### Drug discovery

| Term | Plain meaning |
|---|---|
| **Campaign** | One drug discovery project against one target, from the first experiments to optimized compounds. |
| **Fragment**, **fragment screen** | The first step of a campaign: soak hundreds of tiny molecules into protein crystals and see which ones stick anywhere. Fragments bind weakly but show where binding is possible. |
| **Follow-on compound** | A larger molecule a chemist designs by growing or joining fragments. Binds much more tightly. **Early** and **late** follow-ons are split by when they were made: late ones are further from anything in the fragment screen. |
| **Affinity** | How tightly a ligand binds. Higher affinity means a smaller dose can work. |
| **KD**, **pKD** | KD is the concentration at which half the protein has a ligand bound. pKD is `-log10(KD)`: each +1 means ten times tighter. pKD 7.9 is about 13 nanomolar, a strong binder. |
| **IC50**, **pIC50** | The concentration that blocks half the enzyme's activity. pIC50 works like pKD: higher is more potent. |
| **Fingerprint similarity** (Morgan, Tanimoto) | A score from 0 to 1 for how alike two molecules are, from the chemical groups they share. Used here to tell how far a follow-on is from the fragment screen. |

### Models

| Term | Plain meaning |
|---|---|
| **Co-folding model** | An AI model that predicts the 3D structure of a protein and a ligand together, from the sequence and the SMILES. |
| **AlphaFold3** | Google DeepMind's co-folding model. Its weights are restricted and its training code is not public. |
| **OpenFold3** (OF3) | An open source (Apache 2.0) re-implementation of AlphaFold3 from the AlQuraishi lab and the OpenFold consortium. Free to use commercially and to train yourself, which is why the workshop can fine-tune it. |
| **OpenBind-0** | OpenFold3 with weights trained by the OpenBind consortium. One of the three models compared here. |
| **Boltz-2** | A widely used open co-folding model (MIT licence) that also predicts binding affinity. |
| **Diffusion model** | A model that creates a structure by starting from random noise and removing it step by step. Each run can give a different answer, so a run can draw several *samples* and keep the one it is most confident in. |
| **Recycling** | Feeding a model's own intermediate answer back in to refine it. More recycling steps cost more time. |
| **Steering** (`--use_potentials`) | Boltz-2 option that nudges atoms away from physically impossible arrangements (clashes, broken bonds) while the pose is generated. |
| **Weights**, **checkpoint** | The learned numbers inside a model, saved to a file. Same model, different weights = different predictions. |
| **Fine-tuning** | Continuing to train an already-trained model on a small, specific dataset (here, the EV-A71 2A fragment structures) so it gets better at that case. |
| **Hyperparameter** | A setting chosen before training, such as the learning rate. Ray Tune searches over these. |
| **Learning rate** | How big a step training takes each time it updates the weights. Too small learns nothing, too large breaks what the model already knew. |
| **Loss** | The number training tries to make smaller: how wrong the model is on its training examples. |
| **EMA** (exponential moving average) | A smoothed copy of the weights averaged over recent training steps. Often a bit more stable than the raw weights. |
| **Confidence scores** (pLDDT, ipTM, ranking score) | The model's own estimate of how right it is. ipTM is for the interface between protein and ligand. Useful, but not a substitute for checking against the crystal. |

### Scoring

| Term | Plain meaning |
|---|---|
| **Å** (Ångström) | A tenth of a nanometre. A carbon-carbon bond is about 1.5 Å long. |
| **Superpose**, **align** | Lay the predicted protein on top of the crystal protein so that both are in the same frame before comparing the ligands. Done on the **CA atoms**, one central atom per amino acid. |
| **RMSD** (root mean square deviation) | The typical distance between matching atoms in two structures. **Ligand RMSD** compares only the ligand atoms, after superposing the proteins. |
| **< 2 Å**, **success rate** | The usual cut-off for a correct pose. *Success rate* is the share of poses under it. |
| **lDDT** | A score from 0 to 1 for how well local distances between atoms match the truth. The fine-tune reports it for protein-ligand contacts. |
| **PoseBusters valid** | Passes the PoseBusters checks for a chemically and physically sensible structure. Crystal structures that fail are left out of the curated set. |
| **Crystallographic artefact** | Something in a crystal structure that is an effect of crystal packing or of a poorly resolved region, not real binding. |
| **Curated set** | The binding events kept for the headline numbers: PoseBusters-valid reference, no suspected artefact. |
| **Spearman correlation** | How well two lists agree on *order*, from -1 to 1. 1 means the predicted affinities rank compounds exactly as the lab did; 0 means no relation. |
| **Published** | The OpenBind authors' own result for the same model on the same complexes, used to check that our runs reproduce theirs. |

### Ray and compute

| Term | Plain meaning |
|---|---|
| **Ray** | An open source Python framework for running code across many machines. Anyscale is the managed platform for it. |
| **Ray Data** | Streams a large dataset through processing steps across a cluster. Here, it sends batches of ligands to GPU predictors and the results to CPU scorers. |
| **Ray Train** | Runs one model training job across several GPUs (each holds a copy of the model and they share what they learn). |
| **Ray Tune** | Runs many training or evaluation jobs with different settings side by side, compares them, and can stop the weak ones early. |
| **Ray Serve** | Puts a model behind a web endpoint that scales with traffic. Powers the viewer and live prediction. |
| **ASHA** | A Ray Tune scheduler that checks trials at intervals and stops the ones falling behind, so their GPUs go to the promising ones. |
| **Trial** | One run in a Ray Tune sweep, with one choice of settings. |
| **Actor** | A long-lived Ray worker that keeps state, here a loaded model, between batches. |
| **Autoscaling** | The cluster adds GPU machines when there is work and removes them when there is not. Every worker group here starts at zero. |
| **L4**, **L40S** | NVIDIA data centre GPUs with 24 GB and 48 GB of memory. `g6.2xlarge` is an AWS machine with one L4. |
| **Workspace**, **Job** | Anyscale's interactive development environment, and a one-off run that gets its own cluster and shuts down when finished. |
| **Smoke test** | A tiny run to check that everything is wired up before spending real compute. |

## Data sources and licenses

| Resource | Source | License |
|---|---|---|
| OpenBind EV-A71 2A structure and affinity release | [Zenodo 10.5281/zenodo.22142262](https://zenodo.org/records/22142262) | CC0 |
| OpenBind benchmark results | [OpenBind-Consortium/EV-A71_2A_benchmark](https://github.com/OpenBind-Consortium/EV-A71_2A_benchmark) | Apache 2.0 |
| Boltz-2 | [jwohlwend/boltz](https://github.com/jwohlwend/boltz) | MIT |
| OpenFold3 and OpenBind-0 weights | [aqlaboratory/openfold-3](https://github.com/aqlaboratory/openfold-3) | Apache 2.0 |
| Mpro crystal structures (ASAP Discovery, COVID Moonshot) | [Fragalysis, target CoV-Mpro](https://fragalysis.diamond.ac.uk/) | CC-BY 4.0 |
| Mpro potency (pIC50) | [ASAP x Polaris x OpenADMET challenge](https://huggingface.co/datasets/openadmet/ASAP_Polaris_OpenADMET_challenge) | MIT |
| MSAs for the Mpro targets | [ColabFold MSA server](https://colabfold.mmseqs.com) | |

Reference: *The first OpenBind release: an open experimental structure-affinity
dataset and benchmark for structure-based AI* (bioRxiv, 2026).
