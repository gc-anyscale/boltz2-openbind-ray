"""
Fine-tune OpenFold3 on OpenBind EV-A71 2A fragment structures with Ray Train.

    python prepare_finetune.py                                      # once: build the training set
    python finetune_ray_train.py                                    # 2 L4 GPUs, 150 steps, about 30 min
    python finetune_ray_train.py --num-workers 4 --max-steps 300 --lr 2e-4

Starts from the OpenBind-0 weights and trains on the 66 fragment-screen complexes
written by prepare_finetune.py. OF3's own training stack does the work (data module,
weighted sampler, losses, per-sample gradient clipping, EMA); Ray Train adds:

    TorchTrainer      one worker per GPU, NCCL process group, placement on the cluster
    RayDDPStrategy    Lightning DDP driven by Ray instead of torchrun or SLURM
    ray.train.report  loss terms, protein-ligand lDDT, step time and GPU memory every step
    checkpoint        final weights reported to Ray and written under --storage

The final weights are a plain OF3 state dict, the same format as the OpenBind-0
checkpoint, so `run_openfold predict --inference-ckpt-path <file>` loads them directly.
"""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import yaml

import ray
import ray.train
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from openfold3.entry_points.validator import TrainingExperimentConfig
from prepare_finetune import dataset_paths

SHARED = Path(os.environ.get("AIDDD_SHARED", "/mnt/user_storage/aiddd"))
CONFIG = Path(__file__).parent / "finetune" / "finetune_ev71.yml"
DATA = SHARED / "finetune" / "ev71-2a"
BASE_CKPT = SHARED / "of3_cache" / "of3-ob-2025-06-30-174k.pt"   # OpenBind-0
DATASET = "openbind-ev71-2a"
# Loss terms averaged over workers and reported every step (names as logged by OF3)
REPORTED = ["loss", "diffusion_loss", "distogram_loss", "plddt_loss", "pae_loss",
            "lddt_inter_protein_ligand_step"]


def build_config(args) -> dict:
    """The runner YAML plus everything that depends on paths and flags."""
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["experiment_settings"]["restart_checkpoint_path"] = str(args.base_ckpt)
    cfg["experiment_settings"]["output_dir"] = str(args.storage / args.name / "of3")
    cfg["pl_trainer_args"].update(devices=1, num_nodes=args.num_workers)
    settings = cfg["model_update"]["custom"]["settings"]
    settings["optimizer"]["learning_rate"] = args.lr
    settings["lr_scheduler"]["warmup_no_steps"] = args.warmup_steps
    cfg["model_update"]["custom"]["architecture"]["shared"]["diffusion"]["no_samples"] = args.diffusion_samples
    cfg["data_module_args"]["num_workers"] = args.loader_workers
    cfg["dataset_paths"][DATASET] = dataset_paths(args.data)
    return cfg


def ray_lightning():
    """Ray's Lightning strategy, environment and prepare_trainer for `pytorch_lightning`.

    OF3 is built on `pytorch_lightning`. Ray's helpers subclass `lightning.pytorch` when
    that package is also installed, and the two are not interchangeable, so in that case
    this falls back to equivalent subclasses of the `pytorch_lightning` classes."""
    import pytorch_lightning as pl
    from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, prepare_trainer

    if issubclass(RayDDPStrategy, pl.strategies.DDPStrategy):
        return RayDDPStrategy, RayLightningEnvironment, prepare_trainer

    from pytorch_lightning.plugins.environments import LightningEnvironment

    class Strategy(pl.strategies.DDPStrategy):
        @property
        def root_device(self):
            return ray.train.torch.get_device()

        @property
        def distributed_sampler_kwargs(self):
            return dict(num_replicas=self.world_size, rank=self.global_rank)

    class Environment(LightningEnvironment):
        def world_size(self):
            return ray.train.get_context().get_world_size()

        def global_rank(self):
            return ray.train.get_context().get_world_rank()

        def local_rank(self):
            return ray.train.get_context().get_local_rank()

        def node_rank(self):
            return ray.train.get_context().get_node_rank()

        def set_world_size(self, size):
            pass

        def set_global_rank(self, rank):
            pass

        def teardown(self):
            pass

    return Strategy, Environment, lambda trainer: trainer


def train_loop_per_worker(config: dict):
    """Runs on every GPU worker: OF3's training runner with Ray's Lightning plumbing."""
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.loggers import CSVLogger

    from openfold3.core.utils.callbacks import RankSpecificSeedCallback
    from openfold3.entry_points.experiment_runner import TrainingExperimentRunner

    RayDDPStrategy, RayLightningEnvironment, prepare_trainer = ray_lightning()
    ctx = ray.train.get_context()
    rank = ctx.get_world_rank()

    class ReportEveryStep(pl.Callback):
        """Average the loss over workers and hand it to Ray Train after each optimizer step."""

        def __init__(self):
            self.t0, self.history = time.time(), []

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            logged = {k[len("train/"):]: float(v) for k, v in trainer.callback_metrics.items()
                      if k.startswith("train/") and v.numel() == 1}
            logged["loss"] = float(outputs["loss"] if isinstance(outputs, dict) else outputs)
            # Average the headline terms over workers (each worker trains on a different complex)
            vals = torch.tensor([[logged.get(k, 0.0), k in logged] for k in REPORTED], device=module.device)
            vals = trainer.strategy.reduce(vals, reduce_op="sum")
            averaged = {k: (v / n if n else float("nan")) for k, (v, n) in zip(REPORTED, vals.tolist())}
            now = time.time()
            metrics = {
                "step": trainer.global_step,
                **averaged,
                "lr": trainer.optimizers[0].param_groups[0]["lr"],
                "step_time_s": round(now - self.t0, 2),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
                if torch.cuda.is_available() else 0.0,
                "worker0_pdb_id": batch["pdb_id"][0],
            }
            self.t0 = now
            self.history.append(metrics)
            window = [m["loss"] for m in self.history[-10:]]
            metrics["loss_avg10"] = sum(window) / len(window)
            # Smoothed protein-ligand lDDT: single steps are noisy (one complex per worker,
            # random noise level), so Ray Tune's early stopping compares this instead.
            lddt = [m["lddt_inter_protein_ligand_step"] for m in self.history[-20:]
                    if m["lddt_inter_protein_ligand_step"] == m["lddt_inter_protein_ligand_step"]]
            metrics["lddt_avg20"] = sum(lddt) / len(lddt) if lddt else float("nan")
            if rank == 0:
                print(f"step {metrics['step']:>4}  loss {metrics['loss']:7.3f}  avg10 {metrics['loss_avg10']:7.3f}  "
                      f"distogram {metrics['distogram_loss']:6.3f}  lr {metrics['lr']:.1e}  "
                      f"{metrics['step_time_s']:5.1f} s  {metrics['peak_mem_gb']:.1f} GB", flush=True)
            ray.train.report(metrics)

    report = ReportEveryStep()

    class RayTrainingRunner(TrainingExperimentRunner):
        """OF3's training runner with the Trainer built for Ray Train workers."""

        @property
        def strategy(self):
            # Per-sample gradient clipping in OF3 requires a DDP strategy, even on one worker
            return RayDDPStrategy(find_unused_parameters=False)

        @property
        def callbacks(self):
            # No ModelCheckpoint: it writes a multi-GB file per epoch. The final weights
            # are reported once at the end instead.
            return [RankSpecificSeedCallback(base_seed=self.seed), report]

        @property
        def trainer(self):
            if not hasattr(self, "_trainer"):
                args = self.pl_trainer_args.model_dump(
                    exclude={"deepspeed_config_path", "distributed_timeout", "mpi_plugin",
                             "devices", "num_nodes", "enable_checkpointing"})
                args.update(devices="auto", strategy=self.strategy, plugins=[RayLightningEnvironment()],
                            callbacks=self.callbacks, logger=CSVLogger(self.log_dir), enable_checkpointing=False,
                            enable_progress_bar=False, max_steps=config["max_steps"],
                            default_root_dir=self.output_dir)
                self._trainer = prepare_trainer(pl.Trainer(**args))
            return self._trainer

    runner = RayTrainingRunner(TrainingExperimentConfig.model_validate(config["of3"]))
    runner.setup()      # builds the model and loads the OpenBind-0 weights
    runner.run()        # trainer.fit with OF3's DataModule

    # Final weights: plain OF3 state dict (EMA or raw), reported to Ray as a checkpoint
    module = runner.lightning_module
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = None
        if rank == 0:
            weights = module.ema.state_dict()["params"] if config["export_ema"] else module.model.state_dict()
            torch.save({k: v.detach().cpu() for k, v in weights.items()}, Path(tmp) / "of3_ev71_ft.pt")
            (Path(tmp) / "loss_curve.json").write_text(json.dumps(report.history, indent=1))
            checkpoint = ray.train.Checkpoint.from_directory(tmp)
        last = {k: v for k, v in report.history[-1].items() if k != "worker0_pdb_id"} if report.history else {}
        ray.train.report({**last, "step": module.global_step, "final": True}, checkpoint=checkpoint)


def make_trainer(args, callbacks=()) -> TorchTrainer:
    """The TorchTrainer for one fine-tuning run. tune_finetune.py builds one per trial."""
    of3_config = build_config(args)
    TrainingExperimentConfig.model_validate(of3_config)   # fail fast, before any GPU starts
    scaling = ScalingConfig(
        num_workers=args.num_workers, use_gpu=True, accelerator_type=args.accelerator,
        resources_per_worker={"CPU": min(args.loader_workers + 1, 6)})
    return TorchTrainer(
        train_loop_per_worker,
        train_loop_config={"of3": of3_config, "max_steps": args.max_steps, "export_ema": args.export_ema},
        scaling_config=scaling,
        run_config=RunConfig(name=args.name, storage_path=str(args.storage), callbacks=list(callbacks),
                             checkpoint_config=CheckpointConfig(num_to_keep=1)))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-workers", type=int, default=2, help="GPU workers (one GPU each)")
    ap.add_argument("--max-steps", type=int, default=150, help="optimizer steps (one complex per worker per step)")
    ap.add_argument("--lr", type=float, default=1e-4, help="peak learning rate after warmup")
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--diffusion-samples", type=int, default=8, help="noise levels per complex per step")
    ap.add_argument("--accelerator", default="L4", help="GPU type for the workers")
    ap.add_argument("--loader-workers", type=int, default=4, help="featurization processes per GPU")
    ap.add_argument("--export-ema", action="store_true", help="export EMA weights instead of the raw weights")
    ap.add_argument("--data", type=Path, default=DATA, help="output of prepare_finetune.py")
    ap.add_argument("--base-ckpt", type=Path, default=BASE_CKPT)
    ap.add_argument("--storage", type=Path, default=SHARED / "finetune" / "runs")
    ap.add_argument("--name", default=f"of3-ev71-{time.strftime('%Y%m%d-%H%M%S')}")
    return ap


def check_inputs(args):
    for path in (args.data / "training_cache.json", args.base_ckpt):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run prepare_finetune.py / setup_data.py first.")


def main():
    args = parser().parse_args()
    check_inputs(args)
    start = time.time()
    result = make_trainer(args).fit()

    ckpt_dir = Path(result.checkpoint.path)
    curve = json.loads((ckpt_dir / "loss_curve.json").read_text())
    def mean(rows, key):
        return sum(r[key] for r in rows) / len(rows)

    first, last = curve[:10], curve[-10:]
    print(f"\nDone in {(time.time() - start) / 60:.1f} min: {len(curve)} steps on {args.num_workers} workers")
    for key in REPORTED:
        print(f"  {key:<30} first 10 steps {mean(first, key):.3f} -> last 10 steps {mean(last, key):.3f}")
    print(f"  median step time {sorted(r['step_time_s'] for r in curve)[len(curve) // 2]:.1f} s, "
          f"peak GPU memory {max(r['peak_mem_gb'] for r in curve):.1f} GB")
    print(f"  weights: {ckpt_dir / 'of3_ev71_ft.pt'}")


if __name__ == "__main__":
    main()
