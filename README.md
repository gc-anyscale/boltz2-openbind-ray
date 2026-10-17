# Co-folding at Scale: Boltz-2 and OpenBind-0 on Ray

Reported accuracies for protein-ligand co-folding models are typically measured on curated benchmarks
that resemble the models' training data. Prospective drug discovery presents a different problem: a
single binding pocket, hundreds of related compounds, and chemistry that moves progressively further
from anything the model has seen. This repository evaluates three co-folding models against that
setting, using three antiviral discovery campaigns with 2,147 experimentally determined
protein-ligand complexes, and implements the complete workflow on Ray: distributed inference, pose
scoring, fine-tuning, hyperparameter search and a live prediction service.

The repository accompanies a hands-on workshop at the AI Drug Discovery & Development Summit 2026
(`workshop.ipynb`) and can also be run as a standalone benchmark.

![The workflow end to end: question, crystallographic answer key, prediction, scoring and scale-up](docs/images/workshop.gif)

<sub>Experimental structures and model outputs for EV-A71 2A compound x7259a.</sub>

| Target | Source | Binding events | Compounds | With measured affinity |
|---|---|---|---|---|
| EV-A71 2A protease | OpenBind release | 925 | 699 | 601 (pKD) |
| SARS-CoV-2 main protease (Mpro) | ASAP Discovery and COVID Moonshot, via Fragalysis | 628 | 308 | 164 (pIC50) |
| MERS-CoV main protease (Mpro) | ASAP Discovery, via Fragalysis | 594 | 285 | 142 (pIC50) |

## Contents

- [Background](#background)
- [Study design](#study-design)
- [Results](#results)
- [System design on Ray](#system-design-on-ray)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Runbook](#runbook)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Data sources and licenses](#data-sources-and-licenses)

---

## Background

**Binding pockets.** A protein folds into a three-dimensional structure whose surface contains
concave pockets. In an enzyme, one of these is the active site. The EV-A71 2A protease cleaves the
viral polyprotein into its functional components, and a molecule that blocks its active site
prevents the virus from assembling.

**Pose and affinity.** A bound ligand is held in place by hydrogen bonds, hydrophobic contacts and,
in some cases, coordination to a metal ion. The position and orientation it adopts is its *pose*;
the strength with which it is held is its *affinity*. The two quantities are measured independently,
and a model can predict one correctly and the other incorrectly.

**Crystallographic ground truth.** X-ray crystallography determines the position of every atom in a
protein-ligand complex. The method is slow and costly, which makes a public release of 925 binding
events on a single target an unusually rich benchmark.

**Co-folding.** A co-folding model receives only the protein sequence and the ligand's SMILES string.
It predicts the protein structure and the ligand pose jointly, so the pocket can rearrange to
accommodate the ligand. This matters for EV-A71 2A, where a loop adjacent to the active site is closed
in the apo structure and opens upon binding; rigid-receptor docking fails in precisely this case.

![How a co-folding model predicts a binding pose](docs/images/cofolding_explained.gif)

**Multiple sequence alignments.** Co-folding models also take a multiple sequence alignment (MSA) of
homologous proteins. Residues that co-vary across evolution tend to be in contact in three dimensions,
and this signal accounts for much of the models' folding accuracy. Because every complex of a target
shares the same protein, a single MSA is precomputed per target.

**Pose scoring.** The predicted protein is superposed on the crystal protein, and the root-mean-square
deviation (RMSD) between matched ligand atoms is computed. A ligand RMSD below 2 Å, roughly the length
of a chemical bond, is the conventional threshold for a correct pose. Above it, the ligand typically
occupies the pocket in the wrong orientation, and any structural hypothesis derived from it is
unreliable.

**Discovery campaigns.** A campaign begins with a fragment screen, in which thousands of small
molecules of roughly 15 heavy atoms are soaked into protein crystals. Most do not bind; the few that do
bind weakly but reveal which sub-pockets accept chemistry. Medicinal chemists then elaborate these
fragments over successive design cycles into larger, more potent follow-on compounds.

![From fragment screen to optimized compound](docs/images/campaign_explained.gif)

**Generalization.** Co-folding models are trained on the Protein Data Bank, which samples some protein
families densely and others sparsely. A model may therefore perform well on benchmarks that resemble
its training data and poorly on targets it has not effectively seen. The OpenBind complexes have a
median similarity of approximately 40 to pre-2021 public structures; on that scale, reported
co-folding success rates are 80 to 90 percent above a similarity of 80 and 10 to 40 percent below 50.
Stratifying results by campaign stage probes the same property from the chemical side.

**Affinity prediction.** Pose prediction is a geometric problem, whereas affinity is a free energy that
includes entropic contributions, desolvation of the pocket and protein flexibility. A central finding
of the OpenBind study is negative: a molecular-weight baseline outperformed every structure-based method
in rank correlation. Larger molecules tend to bind more tightly, and a model that captures only this
trend has learned nothing specific about the pocket.

**Target-specific fine-tuning.** Fragment-screen structures show a model what the target pocket looks
like in its ligand-bound state and which chemistry it accommodates. This information transfers to
follow-on compounds, which share the pocket even when they do not share the chemistry. Because every
campaign produces fragment structures early, this is the result of most direct relevance to a
discovery team.

---

## Study design

**Models.**

| Key | Model | Notes |
|---|---|---|
| `boltz2` | Boltz-2 | Predicts pose and binding affinity |
| `openbind0` | OpenFold3 with the OpenBind-0 weights | General-purpose co-folding model |
| `of3ft` | OpenFold3 fine-tuned on the EV-A71 2A fragment screen | Released checkpoint; trained on EV-A71 2A fragments only |
| `of3ray` | OpenFold3 fine-tuned in this repository with Ray Train | Optional; uses the most recent local training run |

**Campaign stages.** Each binding event is assigned to the fragment screen or to the early or late
follow-on phase. For EV-A71 2A, follow-ons are split at the median crystal identifier, which follows
campaign chronology. For the Mpro targets the stages are defined as follows:

| Stage | SARS-CoV-2 Mpro | MERS-CoV Mpro |
|---|---|---|
| Fragment | COVID Moonshot XChem fragment screen, at most 16 heavy atoms | None |
| Early follow-on | ASAP compounds in the Polaris training set | Same |
| Late follow-on | ASAP compounds in the Polaris test set, synthesized later | Same |

Compounds without potency data are staged by crystal order. Binding events more than 8 Å from the
catalytic cysteine are excluded from the curated benchmark.

**Metrics.**

| Term | Definition |
|---|---|
| Ligand RMSD | Heavy-atom RMSD between the predicted and crystal ligand after superposing the protein on its Cα atoms |
| Success rate | Share of binding events with ligand RMSD below 2 Å |
| Published | The OpenBind authors' value for the same method on the same complexes |
| Similarity | Maximum Morgan-fingerprint similarity to any fragment-screen ligand |
| Spearman ρ | Rank correlation between predicted and measured affinity |

---

## Results

All values use the curated set and one sampled pose per ligand.

| Model | EV-A71 2A fragments | EV-A71 2A follow-ons | SARS-CoV-2 fragments | SARS-CoV-2 follow-ons |
|---|---|---|---|---|
| Boltz-2 | 21.6% | 3.2% | 37.5% | 87% |
| OpenBind-0 | 28.4% | 92.7% | 17.5% | 89% |
| OpenFold3 fine-tuned on EV-A71 2A | 81.1%* | 78.9% | 7.5% | 62% |

<sub>*Measured on the model's own training data. All other values are on compounds the model has not seen.</sub>

![Pose accuracy by target and campaign stage](docs/images/pose_accuracy.png)

Panels a and b show the cumulative share of poses within each error threshold; curves that rise
earlier indicate better accuracy, and the dashed line marks the 2 Å cutoff. Panels c and d report the
2 Å success rate by campaign stage.

Three observations stand out. First, accuracy depends strongly on the target: Boltz-2 performs well on
both coronavirus proteases but predicts only 3% of EV-A71 2A follow-on poses correctly. Second,
fine-tuning on fragment structures alone raises follow-on accuracy on EV-A71 2A from 3% (Boltz-2) to
79%, at the cost of lower accuracy on targets the model was not adapted to. Third, a Ray Tune sweep over
Boltz-2 inference settings improves its EV-A71 2A success rate only from 0% to 13.5% at roughly six times
the GPU cost, indicating that model choice and target adaptation matter far more than inference compute
on this target.

**Affinity.** Boltz-2's predicted affinity ranks compounds with Spearman ρ = 0.41 against pKD on
EV-A71 2A (482 ligands), 0.86 against pIC50 on SARS-CoV-2 Mpro (164 ligands) and 0.59 on MERS-CoV Mpro
(142 ligands). On SARS-CoV-2 Mpro the predictions form a weaker and a stronger cluster, and part of the
correlation reflects the separation between them; within the clusters, ρ is 0.65 and 0.38.

![Boltz-2 predicted affinity against measured affinity](docs/images/affinity.png)

**A chemical series through the campaign.** The animation follows one EV-A71 2A series from fragment to
potent follow-on, showing each model's prediction at every step.

![One chemical series from the EV-A71 2A campaign, and how each model follows it](docs/images/campaign.gif)

To reproduce the summary tables for all targets:

```bash
python scripts/compare_targets.py --csv summary.csv
```

The per-complex results are stored as Parquet:

```python
import pandas as pd
df = pd.read_parquet("/mnt/user_storage/aiddd/results/ev71-2a")
df.groupby("stage")[["boltz2_lig_rmsd", "openbind0_lig_rmsd", "of3ft_lig_rmsd"]].median()
```

---

## System design on Ray

Each model runs as its own Ray Data pipeline, and all pipelines share one autoscaling cluster. Scored
results feed a Ray Serve application that provides a leaderboard, a 3D pose viewer and live Boltz-2
prediction.

![Cluster architecture](docs/images/architecture.png)

| Ray library | Role in this repository |
|---|---|
| Ray Data | Streaming pipeline: SMILES validation on CPU, model inference on GPU actors, pose scoring on CPU |
| Ray Train | Distributed fine-tuning of OpenFold3 with its unmodified PyTorch Lightning training loop |
| Ray Tune | Learning-rate sweep with ASHA early stopping, and a sweep over Boltz-2 inference settings |
| Ray Serve | Results viewer and live Boltz-2 prediction, with GPU replicas that autoscale independently |

**One prediction per ligand.** On EV-A71 2A, the 925 binding events correspond to 699 unique ligands,
since some crystals contain two copies. Each ligand is predicted once and scored against every binding
event, which reduces GPU time by about 25%.

**Batched inference.** Each GPU actor processes a batch of ligands in a single `boltz predict` or
`run_openfold predict` call, so model weights are loaded once per batch rather than once per ligand.

**Several predictors per GPU.** A complex of 150 to 300 residues uses less than 5 GB of GPU memory and
cannot saturate an L4 on its own; the GPU also idles while a predictor loads weights and featurizes
inputs on the CPU. Fractional `num_gpus` requests therefore place several predictors on each GPU, so
that one computes while another prepares. Measurements on one L4 (EV-A71 2A, eight ligands per batch):

| Predictors per L4 | GPU activity | Seconds per ligand | GPU memory |
|---|---|---|---|
| OpenFold3, 1 | 32% | 19.9 | 2.9 GB |
| OpenFold3, 2 | 76% | 11.0 | 7.8 GB |
| OpenFold3, 3 | 82% | 9.5 | 10.1 GB |
| Boltz-2, 1 | 75% | 40.8 | 4.8 GB |
| Boltz-2, 2 | 83% | 43.3 | 9.1 GB |

Packing doubles OpenFold3 throughput. Boltz-2 gains little, because its affinity step, which accounts
for more than half of its runtime, already keeps the GPU 98% busy; two predictors per GPU remain the
default because they fill the idle periods during startup.

![Throughput on one L4 with several predictors per GPU](docs/images/throughput.png)

**Concurrent pipelines with fixed GPU shares.** Chaining all models as stages of a single pipeline
forces every stage to run at the pace of the slowest; in one such test, four Boltz-2 GPUs remained idle
for more than ten minutes waiting on an OpenFold3 stage. Instead, each model runs as an independent
pipeline over the validated ligand list, and the driver joins the results before scoring. The GPU budget
(`--max-gpus`) is divided among models according to their measured cost per ligand, and each model
receives a fixed actor pool (`ActorPoolStrategy(size=n, max_tasks_in_flight_per_actor=1)`). A fixed pool
is requested in full, so the cluster scales directly to the budget, and limiting each actor to one
in-flight batch prevents work from queuing behind a busy actor while another sits idle.

| Pipeline design (MERS-CoV Mpro, eight L4 GPUs) | GPU utilization |
|---|---|
| One predictor per GPU, models chained | About 60% across the cluster |
| Packed predictors, models chained | 0% on the four Boltz-2 nodes, 43 to 92% elsewhere |
| Packed predictors, one pipeline per model (default) | 93% on average once all nodes were up |

![GPU utilization per node across the run](docs/images/gpu_cluster.png)

**Further design choices.**

- *Pose scoring.* The predicted protein is superposed on the crystal chain that the ligand actually
  contacts. Many crystals contain two protein copies, and superposing on the wrong one adds 30 to 40 Å
  of error. Ligand RMSD uses element-matched optimal assignment, which handles symmetric groups without
  requiring bond orders.
- *Affinity ensemble.* On 160 EV-A71 2A ligands, a single affinity sample ranks compounds as well as the
  default five (Spearman ρ of 0.37 and 0.36; the two agree at Pearson r = 0.99).
- *Hardware per stage.* Both models fit on a 24 GB L4 for these proteins (150 to 306 residues). For
  larger proteins, add an L40S worker group and pass `--of3-accelerator L40S`.
- *Caching and resumption.* Each ligand's structure (CIF) and metrics (JSON) are written to shared
  storage on completion, so interrupted jobs resume where they stopped.
- *GPU monitoring.* The first predictor on each GPU node samples `nvidia-smi` every five seconds, and
  the job log reports mean utilization per node.

---

## Repository layout

```
.
├── workshop.ipynb              Hands-on workshop notebook (90 minutes)
├── workshop_helpers.py         Plotting, 3D viewer and data-loading helpers for the notebook
├── aiddd_pipeline.py           Ray Data benchmark pipeline: validate, predict, score (one target per run)
├── finetune_ray_train.py       Ray Train fine-tuning of OpenFold3 (TorchTrainer, Lightning DDP)
├── tune_finetune.py            Ray Tune learning-rate sweep over complete Ray Train runs
├── tune_boltz.py               Ray Tune sweep over Boltz-2 inference settings
├── serve_viewer.py             Ray Serve application: results viewer and live prediction
├── job.yaml                    Anyscale Job definition with autoscaling L4 and CPU worker groups
├── service.yaml                Anyscale Service definition for sharing the viewer
├── configs/
│   └── finetune_ev71.yml       OpenFold3 training settings used by the fine-tune
├── scripts/
│   ├── setup_data.py           One-time download of data, published benchmark and model weights
│   ├── prepare_mpro.py         One-time construction of the SARS-CoV-2 and MERS-CoV Mpro targets
│   ├── prepare_finetune.py     Builds the OpenFold3 training set from the EV-A71 2A fragment screen
│   ├── compare_targets.py      Summary table across targets, models and campaign stages
│   └── check_serve.py          End-to-end check of the Serve application
├── image/Containerfile         Container image: Ray 2.59, Boltz-2, OpenFold3
└── docs/
    ├── images/                 Figures and animations used in this README and the notebook
    └── figures/                Scripts that regenerate every figure from the benchmark results
```

---

## Getting started

Assuming an Anyscale Workspace configured as in [Step 1](#step-1-workspace-and-image), four commands
reproduce the study:

```bash
python scripts/setup_data.py                            # 15 to 20 min, once
python aiddd_pipeline.py --limit 8 --max-gpus 2 \
  --out /mnt/user_storage/aiddd/results/smoke           # 15 min, verifies the stack
anyscale job submit -f job.yaml --name aiddd-ev71       # 1 to 2 h, full benchmark
serve run serve_viewer:app                              # viewer and live prediction
```

To follow the workshop instead, open `workshop.ipynb` in the Workspace and run the cells in order.

**Workshop format.** Steps 2, 2b and 4 should be completed before the session. The 90-minute session
then proceeds as follows:

| Minutes | Activity |
|---|---|
| 0 to 10 | Workspace setup; submit the smoke test while nodes start |
| 10 to 25 | Smoke-test results and interpretation of the scoring output |
| 25 to 45 | Full benchmark: campaign-stage and cross-target comparison |
| 45 to 60 | 3D viewer and pose overlays |
| 60 to 80 | Ray Train fine-tuning on two L4 GPUs |
| 80 to 90 | Ray Serve: live prediction for a molecule proposed by the audience |

---

## Runbook

Wall-clock times refer to the cluster defined in `job.yaml`.

| Step | Task | Time |
|---|---|---|
| 1 | Workspace and image | 5 min with a prebuilt image |
| 2 | Download data and weights | 15 to 20 min, once |
| 2b | Build the Mpro targets | About 10 min, once |
| 3 | Smoke test: eight ligands, three models | 15 to 20 min, mostly node startup |
| 4 | Full benchmark: one target, three models | 1 to 2 h on eight L4 GPUs |
| 5 | Viewer and live prediction | 5 min |
| 6 | Fine-tuning with Ray Train: 150 steps on two L4 GPUs | About 30 min |
| 7 | Sweeps with Ray Tune | 35 to 60 min |

### Step 1. Workspace and image

| Setting | Value |
|---|---|
| Container image | `anyscale/image/aiddd-boltz2-of3:4` |
| Head node | `m5.2xlarge`, CPU only, so that notebooks never hold a GPU |
| Worker group | `g6.xlarge` or `g6.2xlarge` (one L4), minimum 0, maximum 4 |

Clone this repository into the Workspace. To build the image yourself, run the following command and
update `image_uri` in `job.yaml` and `service.yaml` to the version it reports:

```bash
anyscale image build -f image/Containerfile -n aiddd-boltz2-of3 -r 2.59.0
```

> **Python version.** Boltz-2 pins a SciPy release that does not build on Python 3.13. The image also
> contains a separate OpenFold3 0.4.3 installation at `/opt/of3-legacy`, because the released
> fine-tuned checkpoint loads only with that version.

### Step 2. Download data and weights

```bash
anyscale job submit -f job.yaml --name aiddd-setup -- python scripts/setup_data.py
```

All data is stored under `/mnt/user_storage/aiddd`, which both Workspaces and Jobs can read:

```
/mnt/user_storage/aiddd/
  raw/OpenBind_EV-A71_2A/      structures, metadata CSV, FASTA, MSA        (500 MB)
  EV-A71_2A_benchmark/         published OpenBind results                  (11 MB)
  boltz_cache/                 Boltz-2 weights and chemical component data (6 GB)
  of3_cache/                   OpenBind-0 and fine-tuned OF3 checkpoints   (9 GB)
```

The script skips anything already present and is safe to rerun. Pass `--skip-of3` to download Boltz-2
only.

### Step 2b. Build the Mpro targets

```bash
anyscale job submit -f job.yaml --name aiddd-mpro-setup -- python scripts/prepare_mpro.py
```

The script downloads the ASAP Discovery and COVID Moonshot structures from Fragalysis, joins the ASAP
potency data, builds one MSA per protein and writes each target in the layout of the OpenBind release.

### Step 3. Smoke test

Eight ligands (two fragments and six follow-ons) through all three models:

```bash
anyscale job submit -f job.yaml --name aiddd-smoke -- \
  python aiddd_pipeline.py --models boltz2,openbind0,of3ft \
    --limit 8 --max-gpus 2 --out /mnt/user_storage/aiddd/results/smoke
```

The job log ends with a summary such as:

```
Done in N min. 10 complexes scored -> /mnt/user_storage/aiddd/results/smoke
  Boltz-2: {'ok': 8}
  OpenBind-0 (OF3): {'ok': 8}
  OF3 fine-tuned on EV-A71 2A: {'ok': 8}
```

A status of `cached` also indicates success: an earlier run had already predicted that ligand. For any
`failed` entries, the first error is printed below the summary.

### Step 4. Full benchmark

```bash
anyscale job submit -f job.yaml --name aiddd-ev71        # EV-A71 2A, the job.yaml default
anyscale job submit -f job.yaml --name aiddd-sars2 -- \
  python aiddd_pipeline.py --target sars2-mpro --models boltz2,openbind0,of3ft --max-gpus 8
anyscale job submit -f job.yaml --name aiddd-mers -- \
  python aiddd_pipeline.py --target mers-mpro --models boltz2,openbind0,of3ft --max-gpus 8
```

Each job runs on its own cluster, so the three targets can be processed concurrently. Predictions are
cached per ligand, so a resubmitted job predicts only what is missing. While a job runs, the Metrics
page of the Ray Dashboard shows GPU nodes joining the cluster and GPU utilization per node.

### Step 5. Viewer and live prediction

```bash
python scripts/check_serve.py                                      # end-to-end check
AIDDD_RESULTS=/mnt/user_storage/aiddd/results/ev71-2a serve run serve_viewer:app
```

Open port 8000 from the Workspace **Ports** tab. The application provides:

- **Leaderboard.** Every crystal complex, sorted by ligand RMSD, with a tab per model.
- **3D viewer.** Selecting a row overlays the predicted ligand (green) on the crystal ligand (orange).
- **Live prediction.** A SMILES string submitted on the page is sent to a Boltz-2 replica on an L4,
  which returns the pose, ligand ipTM, predicted affinity and binder probability. A new molecule takes
  about one minute; repeated molecules are served from a cache.

The Boltz-2 deployment autoscales from one to four L4 replicas. To share the viewer with a room, deploy
it as an Anyscale Service:

```bash
anyscale service deploy -f service.yaml          # prints the URL to share
anyscale service terminate -n aiddd-viewer       # after the session
```

### Step 6. Fine-tuning with Ray Train

The `of3ft` model evaluates a released checkpoint fine-tuned on EV-A71 2A fragment structures. This
step reproduces how such a checkpoint is produced.

**Build the training set** (CPU, about 30 seconds, once):

```bash
python scripts/prepare_finetune.py --check
```

The script selects fragment-screen complexes with a valid reference pose, no suspected artefact and no
covalent ligand, keeping one complex per compound (66 in total), and writes them in the format read by
OpenFold3's training data loader. The `--check` flag loads a sample back through OpenFold3's
`WeightedPDBDataset` to validate the format.

**Train** (two L4 GPUs, about 30 minutes including node startup):

```bash
anyscale job submit -f job.yaml --name aiddd-finetune -- \
  python finetune_ray_train.py --num-workers 2 --max-steps 150
```

```
Done in 27.7 min: 150 steps on 2 workers
  loss                           first 10 steps 0.138 -> last 10 steps 0.113
  lddt_inter_protein_ligand_step first 10 steps 0.434 -> last 10 steps 0.576
  median step time 9.2 s, peak GPU memory 13.6 GB
  weights: /mnt/user_storage/aiddd/finetune/runs/<name>/checkpoint_.../of3_ev71_ft.pt
```

| Component | Role |
|---|---|
| `TorchTrainer` with `ScalingConfig(num_workers=N, use_gpu=True)` | One training worker per L4 in an NCCL process group; nodes scale from zero |
| `RayDDPStrategy`, `RayLightningEnvironment` | Run OpenFold3's unmodified Lightning loop under Ray instead of torchrun or SLURM |
| `ray.train.report` | Reports loss terms, learning rate, step time and GPU memory at every step |
| Final checkpoint | A plain OpenFold3 state dictionary, in the same format as OpenBind-0 |

Training settings are in `configs/finetune_ev71.yml`, a standard OpenFold3 runner configuration. The
defaults fit one 24 GB L4 per worker: a 192-token budget holds a complete complex without cropping, and
eight diffusion noise levels are sampled per step instead of 48.

| Flag | Default | Description |
|---|---|---|
| `--num-workers` | 2 | Number of GPUs; one complex per GPU per step |
| `--max-steps` | 150 | Optimizer steps |
| `--lr` | 1e-4 | Peak learning rate, reached after `--warmup-steps` (10) |
| `--diffusion-samples` | 8 | Noise levels per complex per step |
| `--export-ema` | off | Export the EMA weights instead of the raw weights |

![Ray Train fine-tuning curves](docs/images/finetune.png)

Because OpenBind-0 already predicts these fragment complexes well, the initial loss is low and
individual steps are noisy: each worker sees a different complex, and the diffusion loss depends on a
randomly drawn noise level. Excluding the first step, which includes kernel compilation, the training
loss is nearly flat over 150 steps (0.120 to 0.117 across the first and last 20 steps). The protein-ligand
lDDT of a short structure rollout during training is a clearer signal, rising from 0.44 to 0.57. This is a
demonstration-scale run; the released `of3ft` checkpoint was trained for 1,000 steps on eight GPUs.

**Benchmark the new weights.** The `of3ray` model uses the most recent weights from this step, or the
file given by `--of3ray-ckpt`:

```bash
anyscale job submit -f job.yaml --name aiddd-of3ray -- \
  python aiddd_pipeline.py --models of3ray,of3ft,openbind0
```

Because predictions are cached per ligand, delete `/mnt/user_storage/aiddd/predictions/of3ray` after
retraining.

### Step 7. Sweeps with Ray Tune

```bash
anyscale job submit -f job.yaml --name tune-ft    -- python tune_finetune.py   # about 35 min on four L4 GPUs
anyscale job submit -f job.yaml --name tune-boltz -- python tune_boltz.py      # about 60 min on eight L4 GPUs
```

`tune_finetune.py` runs each learning rate as a complete Ray Train trial, stops underperforming trials
with the ASHA scheduler and evaluates the best weights on held-out early follow-on complexes.
`tune_boltz.py` evaluates combinations of recycling steps, diffusion samples and physics-based steering
for Boltz-2, one L4 per trial.

---

## Configuration reference

### `aiddd_pipeline.py`

| Flag | Default | Description |
|---|---|---|
| `--target` | `ev71-2a` | `ev71-2a`, `sars2-mpro` or `mers-mpro` |
| `--models` | `boltz2,openbind0` | Any of `boltz2`, `openbind0`, `of3ft`, `of3ray` |
| `--limit` | all | First N ligands, a fixed mix of fragments and follow-ons |
| `--batch-size` | 8 | Ligands per GPU call; 8 to 16 works well |
| `--boltz-per-gpu` | 2 | Boltz-2 predictors sharing one GPU |
| `--of3-per-gpu` | 3 | OpenFold3 predictors sharing one GPU |
| `--max-gpus` | 8 | GPUs for the whole pipeline, divided among models by cost |
| `--accelerator` | `L4` | GPU type for Boltz-2, and for OpenFold3 unless overridden |
| `--of3-accelerator` | same as `--accelerator` | GPU type for the OpenFold3 stages |
| `--diffusion-samples` | 1 | Poses sampled per ligand |
| `--affinity-samples` | 5 | Boltz-2 affinity ensemble size, cached separately |
| `--affinity-only` | off | Only ligands with a measured affinity |
| `--of3ray-ckpt` | latest Ray Train run | Weights for the `of3ray` model |
| `--out` | `$AIDDD_SHARED/results/<target>` | Output Parquet directory |

### Environment variables

| Variable | Default | Used by |
|---|---|---|
| `AIDDD_SHARED` | `/mnt/user_storage/aiddd` | All scripts: data, weights, predictions and results |
| `AIDDD_RESULTS` | `$AIDDD_SHARED/results/ev71-2a` | `serve_viewer.py` |

---

## Troubleshooting

| Symptom | Cause and remedy |
|---|---|
| `Missing /mnt/user_storage/aiddd/...` | Data has not been downloaded. Run Step 2 or Step 2b |
| Job remains at `pending=1` actors for several minutes | GPU nodes are starting; the first node typically takes three to six minutes |
| `InsufficientInstanceCapacity` | The cloud has no capacity for that GPU type. `enable_cross_zone_scaling` retries other zones; if all fail, drop `--of3-accelerator L40S` so that every stage runs on L4 |
| `Default checkpoint ... not found` from OpenFold3 | `of3_cache` is incomplete. Rerun `python scripts/setup_data.py` |
| `of3ft` fails with missing or unexpected state-dictionary keys | The job is not using the project image, so `/opt/of3-legacy` is absent. Check `image_uri` |
| CUDA out of memory in an OpenFold3 stage | Lower `--batch-size`, or add an L40S group and pass `--of3-accelerator L40S` |
| Viewer shows `not scored yet` | `AIDDD_RESULTS` points to a run that does not include that model |
| Live prediction never returns | No L4 is available to the Workspace. Add an L4 worker group |
| 3D view in the notebook is empty | The notebook is not trusted. Run the cell again, or run `jupyter trust workshop.ipynb` |

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
| Ray diagrams in the notebook | [Ray documentation](https://docs.ray.io) | Apache 2.0 |

**Reference.** *The first OpenBind release: an open experimental structure-affinity dataset and
benchmark for structure-based AI* (bioRxiv, 2026).
