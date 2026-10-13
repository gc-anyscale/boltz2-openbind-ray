# Co-folding at scale: Boltz-2 and OpenBind-0 on Ray

How well do modern co-folding models predict where and how a small molecule binds,
on real drug discovery campaigns rather than curated benchmarks? This repository
answers that question end to end on three antiviral targets, each with hundreds of
experimentally solved protein-ligand complexes that follow a campaign from the
first fragment screen through to optimized follow-on compounds:

| Target | Source | Binding events | Compounds | With affinity |
|---|---|---|---|---|
| EV-A71 2A protease | OpenBind release | 925 | 699 | 601 (pKD) |
| SARS-CoV-2 main protease (Mpro) | ASAP Discovery and COVID Moonshot via Fragalysis | 628 | 308 | 164 (pIC50) |
| MERS-CoV main protease (Mpro) | ASAP Discovery via Fragalysis | 594 | 285 | 142 (pIC50) |

**The workshop in 30 seconds.** A drug works by fitting into a pocket on a protein.
Given only the protein's amino-acid sequence and the drug's chemical formula (SMILES),
a co-folding model predicts the 3D structure of the two together: where the drug sits
and how it is oriented (its *pose*). We check every prediction against an X-ray
crystal structure, which shows the true pose, and repeat this for every complex on a
GPU cluster with Ray.

![The workshop end to end: question, crystal answer key, prediction, scoring, scale-up, workflow](docs/images/workshop.gif)

<sub>Real structures and real model outputs for EV-A71 2A compound x7259a. The
noise-to-pose motion in step 3 illustrates how diffusion models work; the final
poses are the models' actual predictions.</sub>

**Fragments and follow-ons.** A drug discovery campaign starts with a *fragment
screen*: very small molecules soaked into protein crystals to find where anything
binds. Chemists then grow the best fragments into larger *follow-on* compounds that
bind much more tightly. Follow-ons contain chemistry the fragment screen never showed,
so they test whether a model generalizes. Here is one real series from the EV-A71 2A
campaign:

![A fragment grown into a potent follow-on compound, and how each model does at each stage](docs/images/campaign.gif)

You will:

1. Run **Boltz-2**, **OpenBind-0**, and a **target fine-tuned OpenFold3** checkpoint over
   every compound with **Ray Data**, on a GPU cluster that scales from zero.
2. **Score** every predicted pose against its crystal structure (ligand RMSD) and
   compare with the published OpenBind numbers.
3. Compare accuracy **across targets** and as compounds **move away from the fragment screen**.
4. Browse the results in an interactive **3D viewer** and run **live predictions**
   for new molecules with **Ray Serve**.
5. **Fine-tune** OpenFold3 on the fragment screen with **Ray Train**, and benchmark
   your own weights on the follow-on compounds.

![Workshop workflow](docs/images/workflow.png)

---

## Contents

- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Runbook](#runbook)
  - [Step 1. Workspace and image](#step-1-workspace-and-image)
  - [Step 2. Download data and weights](#step-2-download-data-and-weights)
  - [Step 2b. Build the Mpro targets](#step-2b-build-the-mpro-targets)
  - [Step 3. Smoke test](#step-3-smoke-test)
  - [Step 4. Full benchmark](#step-4-full-benchmark)
  - [Step 5. Read the results](#step-5-read-the-results)
  - [Step 6. Viewer and live prediction](#step-6-viewer-and-live-prediction)
  - [Step 7. Fine-tune with Ray Train](#step-7-fine-tune-with-ray-train)
- [How it works](#how-it-works)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Data sources and licenses](#data-sources-and-licenses)

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
| `job.yaml` | Anyscale Job definition with autoscaling L4 and CPU worker groups |
| `image/Containerfile` | Container image: Ray 2.59, Python 3.12, Boltz-2, OpenFold3 |
| `docs/images/` | Figures in this README, drawn from the runs described here |
| `docs/figures/` | Scripts that redraw every figure and GIF from the results under `/mnt/user_storage/aiddd` |

---

## Runbook

Times are wall clock on the cluster in `job.yaml`.

| Step | What | Time |
|---|---|---|
| 1 | Workspace and image | 5 min (image prebuilt) |
| 2 | Download data and weights | 15 to 20 min, once |
| 2b | Build the Mpro targets | about 10 min, once |
| 3 | Smoke test, 8 ligands, all three models | 15 to 20 min, mostly node startup |
| 4 | Full benchmark, one target, three models | about 1 to 2 h on 8 L4, longer for larger proteins |
| 5 | Read the results | 5 min |
| 6 | Viewer and live prediction | 5 min |
| 7 | Fine-tune with Ray Train, 150 steps on 2 L4 | about 30 min |

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
not a held-out test; its follow-on and SARS-CoV-2 scores are.

**Reference results** from our runs (share of poses under 2 A on the curated set;
your numbers should be close, since each model samples one pose per ligand):

| Model | EV-A71 2A fragments | EV-A71 2A follow-ons | SARS-CoV-2 Mpro fragments | SARS-CoV-2 Mpro follow-ons |
|---|---|---|---|---|
| Boltz-2 | 21.6% | 3.2% | 37.5% | 87% |
| OpenBind-0 | 28.4% | 92.7% | 17.5% | 89% |
| OF3 fine-tuned on EV-A71 2A | 81.1% | 78.9% | 7.5% | 62% |

On EV-A71 2A follow-ons the published numbers for the same complexes are 7.1%
(Boltz-2) and 63.4% (fine-tuned OF3). Boltz-2's predicted affinity ranks compounds
with Spearman 0.41 against pKD on EV-A71 2A (482 ligands) and 0.86 against pIC50 on
SARS-CoV-2 Mpro (164 ligands).

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
python compare_targets.py --csv summary.csv
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

**1. Build the training set** (CPU, about 30 s, once):

```bash
python prepare_finetune.py --check
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
