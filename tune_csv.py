"""Ray Tune driver for CSV-driven source-model training.

Reuses tableshift/configs/hparams.py:search_space[model] (so tabtransformer
gets lr/n_epochs/weight_decay/ff_dropout/attn_dropout/dim/depth/heads),
optimizes *validation* balanced_accuracy via HyperOpt+ASHA on a
CSVTabularDataset, then refits the best config and drops checkpoint.pt +
metrics.json into models/<exp>/<arch>/.

Picks balanced_accuracy on validation as the search metric (not OOD) to
avoid leaking the OOD split into model selection.
"""

import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
import ray
from ray import tune
from ray.air import session
from ray.air.config import RunConfig
from ray.tune.schedulers import ASHAScheduler

try:
    from ray.tune.search.hyperopt import HyperOptSearch
    _HAS_HYPEROPT = True
except ImportError:
    _HAS_HYPEROPT = False

from tableshift.configs.hparams import search_space
from tableshift.core.csv_dataset import CSVTabularDataset
from tableshift.models.utils import get_estimator
from tableshift.models.default_hparams import get_default_config


SUPPORTED_EXPERIMENTS = (
    "acsfoodstamps",
    "acsunemployment",
    "assistments",
    "brfss_diabetes",
    "nhanes_lead",
    "physionet",
)

ARCH_DIRNAME = {
    "mlp": "mlp",
    "tabtransformer": "tabtrans",
    "ft_transformer": "fttrans",
}


def _build_config(model, dset, sample_cfg, batch_size, max_epochs):
    cfg = get_default_config(model, dset)
    cfg.update(sample_cfg)
    if batch_size is not None:
        cfg["batch_size"] = batch_size
    cfg["n_epochs"] = min(int(cfg["n_epochs"]), int(max_epochs))
    return cfg


def _coerce_for_json(v):
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def trainable(sample_cfg, *, experiment, data_dir, model,
              batch_size, max_epochs):
    """Per-trial fn. Manual epoch loop so we can report each epoch to ASHA."""
    torch.manual_seed(0)
    dset = CSVTabularDataset(name=experiment, data_dir=data_dir)
    cfg = _build_config(model, dset, sample_cfg, batch_size, max_epochs)
    cfg["exp"] = experiment

    estimator = get_estimator(model, **cfg)
    if hasattr(estimator, "config") and isinstance(estimator.config, dict):
        estimator.config["exp"] = experiment

    device = (f"cuda:{torch.cuda.current_device()}"
              if torch.cuda.is_available() else "cpu")
    estimator.to(device)

    bs = cfg["batch_size"]
    train_loader = dset.get_dataloader("train", batch_size=bs, shuffle=True)
    val_loader = dset.get_dataloader("validation",
                                     batch_size=bs, shuffle=False)
    eval_loaders = {"validation": val_loader}
    loss_fn = cfg["criterion"]

    for epoch in range(1, int(cfg["n_epochs"]) + 1):
        estimator.train_epoch(train_loaders={"train": train_loader},
                              loss_fn=loss_fn, device=device,
                              eval_loaders=eval_loaders,
                              max_examples_per_epoch=dset.n_train)
        estimator.evaluate(eval_loaders, device=device)
        bal = estimator._last_eval_metrics["validation"]["balanced_accuracy"]
        acc = estimator._last_eval_metrics["validation"]["accuracy"]
        f1 = estimator._last_eval_metrics["validation"]["f1"]
        session.report({"balanced_acc": float(bal),
                        "accuracy": float(acc),
                        "f1": float(f1),
                        "epoch": epoch})


def refit_best(best_cfg, experiment, data_dir, model, batch_size,
               max_epochs, ckpt_root):
    """Train once more with the winning config and persist checkpoint."""
    from tableshift.models.training import train as _train

    torch.manual_seed(0)
    dset = CSVTabularDataset(name=experiment, data_dir=data_dir)
    cfg = _build_config(model, dset, best_cfg, batch_size, max_epochs)
    cfg["exp"] = experiment

    estimator = get_estimator(model, **cfg)
    if hasattr(estimator, "config") and isinstance(estimator.config, dict):
        estimator.config["exp"] = experiment

    banner = (f" REFIT | experiment={experiment} | model={model} | "
              f"n_epochs={cfg['n_epochs']} | batch_size={cfg['batch_size']} | "
              f"lr={cfg.get('lr')} | wd={cfg.get('weight_decay')} ")
    print("=" * len(banner))
    print(banner)
    print("=" * len(banner), flush=True)

    estimator = _train(estimator, dset, config=cfg)

    arch_dir = ARCH_DIRNAME[model]
    out_dir = os.path.join(ckpt_root, experiment, arch_dir)
    os.makedirs(out_dir, exist_ok=True)
    src_ckpt = os.path.join("model", "checkpoint.pt")
    if os.path.exists(src_ckpt):
        dst = os.path.join(out_dir, "checkpoint.pt")
        shutil.move(src_ckpt, dst)
        print(f"[save] checkpoint -> {dst}", flush=True)

    metrics = getattr(estimator, "_last_eval_metrics", None) or {}
    payload = {
        "experiment": experiment,
        "model": model,
        "best_config": {k: _coerce_for_json(v) for k, v in best_cfg.items()},
        "n_epochs": cfg["n_epochs"],
        "batch_size": cfg["batch_size"],
        "n_train": dset.n_train,
        "splits": {k: metrics[k] for k in ("validation", "id_test", "ood_test")
                   if k in metrics},
    }
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[save] metrics    -> {metrics_path}", flush=True)


def main(args):
    if args.experiment not in SUPPORTED_EXPERIMENTS:
        raise ValueError(f"unsupported experiment {args.experiment!r}")
    if args.model not in search_space:
        raise ValueError(f"no search space for model {args.model!r}")
    space = dict(search_space[args.model])
    if args.batch_size is None:
        space["batch_size"] = tune.choice([128, 256, 512, 1024])
        print(f"[tune] searching batch_size over {[128, 256, 512, 1024]}",
              flush=True)
    else:
        print(f"[tune] batch_size fixed to {args.batch_size}", flush=True)

    ray.init(ignore_reinit_error=True)

    grace = min(3, args.max_epochs)
    scheduler = ASHAScheduler(max_t=args.max_epochs,
                              grace_period=grace,
                              reduction_factor=3)
    if _HAS_HYPEROPT:
        searcher = HyperOptSearch()
    else:
        print("[tune] hyperopt not installed; falling back to random search.",
              flush=True)
        searcher = None

    trainable_fn = tune.with_parameters(
        trainable,
        experiment=args.experiment,
        data_dir=args.data_dir,
        model=args.model,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
    )
    trainable_with_resources = tune.with_resources(
        trainable_fn,
        {"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
    )

    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=space,
        tune_config=tune.TuneConfig(
            metric="balanced_acc",
            mode="max",
            num_samples=args.num_samples,
            scheduler=scheduler,
            search_alg=searcher,
        ),
        run_config=RunConfig(
            name=f"tune_{args.experiment}_{args.model}",
            local_dir=args.local_dir,
            verbose=1,
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(metric="balanced_acc", mode="max")
    print(f"[tune] best balanced_acc={best.metrics['balanced_acc']:.4f} "
          f"(acc={best.metrics.get('accuracy'):.4f}, "
          f"f1={best.metrics.get('f1'):.4f}, "
          f"epoch={best.metrics.get('epoch')})", flush=True)
    print(f"[tune] best config={best.config}", flush=True)
    print(f"[tune] best logdir={best.log_dir}", flush=True)

    if args.refit:
        refit_best(best.config, args.experiment, args.data_dir, args.model,
                   args.batch_size, args.max_epochs, args.ckpt_root)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",
                   default="/data/xli/skw/Advanced-ML/data/TableShift")
    p.add_argument("--ckpt_root", default="./models")
    p.add_argument("--local_dir", default=os.path.abspath("./ray_results"))
    p.add_argument("--experiment", default="assistments",
                   choices=SUPPORTED_EXPERIMENTS)
    p.add_argument("--model", default="tabtransformer")
    p.add_argument("--num_samples", type=int, default=16,
                   help="Number of trials to run.")
    p.add_argument("--max_epochs", type=int, default=20,
                   help="Cap n_epochs sampled from the search space; "
                        "also used as ASHA max_t.")
    p.add_argument("--batch_size", type=int, default=None,
                   help="If set, fixes batch size across trials and refit. "
                        "If omitted, batch_size is searched alongside other "
                        "hparams.")
    p.add_argument("--cpus_per_trial", type=float, default=4)
    p.add_argument("--gpus_per_trial", type=float, default=0.5,
                   help="Fractional GPU per trial (e.g. 0.5 fits 2 trials/GPU).")
    p.add_argument("--refit", action="store_true", default=True)
    p.add_argument("--no_refit", dest="refit", action="store_false")
    args = p.parse_args()
    main(args)
    print(f"[exit] main returned at {time.strftime('%H:%M:%S')}, "
          f"forcing os._exit(0) to skip slow interpreter shutdown",
          flush=True)
    os._exit(0)
