"""CSV-driven entry point for PFT3A test-time adaptation.

Loads a source-trained checkpoint and adapts it on the OOD-test split via
PFT3A (online entropy + KL minimization on unlabeled OOD batches). No
further supervised training on the source-train split is performed.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from tqdm import tqdm

from tableshift.core.csv_dataset import CSVTabularDataset
from tableshift.models.utils import get_estimator
from tableshift.models.default_hparams import get_default_config
from tableshift.models.torchutils import unpack_batch
from tableshift.PFT3A_src.PFT3A import PFT3A


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


def main(experiment, data_dir, model, ckpt_root, n_epochs, batch_size, debug):
    if debug:
        print("[INFO] running in debug mode.")
        experiment = "_debug"

    if experiment not in SUPPORTED_EXPERIMENTS:
        raise ValueError(
            f"unsupported experiment {experiment!r}; "
            f"expected one of {SUPPORTED_EXPERIMENTS}")

    dset = CSVTabularDataset(name=experiment, data_dir=data_dir)

    if model not in ARCH_DIRNAME:
        raise ValueError(f"unsupported model {model!r}")
    ckpt_dir = f"{ckpt_root.rstrip('/')}/{experiment}/{ARCH_DIRNAME[model]}"
    ckpt_path = f"{ckpt_dir}/checkpoint.pt"

    config = get_default_config(model, dset)

    # Overlay tuned hparams from metrics.json so the architecture matches the
    # checkpoint produced by tune_csv.py / source_model_training_csv.py. Skip
    # n_epochs because here it refers to TTA passes, not source training.
    metrics_path = os.path.join(ckpt_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            best_config = json.load(f).get("best_config", {})
        for k, v in best_config.items():
            if k == "n_epochs":
                continue
            config[k] = v
        print(f"[load] applied best_config from {metrics_path}: {best_config}")

    config["exp"] = experiment
    config["n_epochs"] = n_epochs
    if batch_size is not None:
        config["batch_size"] = batch_size
    estimator = get_estimator(model, **config)
    if hasattr(estimator, "config") and isinstance(estimator.config, dict):
        estimator.config["exp"] = experiment

    print(ckpt_path)
    para, _ = torch.load(ckpt_path)
    estimator.load_state_dict(para)
    estimator._csv_source_y_prior = dset.source_y_prior

    device = (f"cuda:{torch.cuda.current_device()}"
              if torch.cuda.is_available() else "cpu")
    estimator.to(device)
    estimator.train()

    bs = config["batch_size"]
    ood_loader = dset.get_dataloader("ood_test", batch_size=bs, shuffle=False)

    p1 = float(dset.source_y_prior[1])
    source_y = torch.tensor([1.0 - p1, p1]).to(device)
    pft3a = PFT3A(estimator, torch.optim.SGD, prior=source_y,
                  lr_list=[1e-5, 5e-4, 1e-4], device=device)

    init_x, _, _, _ = unpack_batch(next(iter(ood_loader)))
    init_x = init_x.float().to(device)
    pft3a.get_prior(init_x)

    banner = (f" TTA | experiment={experiment} | model={model} | "
              f"n_epochs={n_epochs} | batch_size={bs} | "
              f"ckpt={ckpt_path} ")
    print("=" * len(banner))
    print(banner)
    print("=" * len(banner), flush=True)

    preds, labels = [], []
    total_used = 0
    total_seen = 0
    with torch.enable_grad():
        for epoch in range(1, n_epochs + 1):
            preds, labels = [], []
            ep_used = 0
            ep_seen = 0
            for batch in tqdm(ood_loader,
                              desc=f"{experiment}:pft3a:tta-ep{epoch}"):
                x, y, _, _ = unpack_batch(batch)
                x = x.float().to(device)
                y = y.float().to(device)
                out = pft3a(x, y)
                preds.append(out.detach().cpu().numpy())
                labels.append(y.detach().cpu().numpy())
                ep_used += pft3a.last_remain_count
                ep_seen += pft3a.last_batch_size
            print(f"[tta] epoch {epoch}: updated on {ep_used}/{ep_seen} "
                  f"OOD samples ({ep_used / max(ep_seen, 1):.1%})",
                  flush=True)
            total_used += ep_used
            total_seen += ep_seen

    pred_arr = (np.concatenate(preds).reshape(-1) >= 0.5).astype(int)
    label_arr = np.concatenate(labels).reshape(-1).astype(int)
    acc = accuracy_score(label_arr, pred_arr)
    bal_acc = balanced_accuracy_score(label_arr, pred_arr)
    f1 = f1_score(label_arr, pred_arr, zero_division=0)
    print(f"[tta] ood_test acc={acc:.4f} "
          f"balanced_acc={bal_acc:.4f} f1={f1:.4f}", flush=True)
    print(f"[tta] total updates on {total_used}/{total_seen} OOD samples "
          f"across {n_epochs} epoch(s) "
          f"({total_used / max(total_seen, 1):.1%})", flush=True)

    arch_dir = ARCH_DIRNAME[model]
    out_dir = os.path.join(ckpt_root, experiment, arch_dir)
    os.makedirs(out_dir, exist_ok=True)
    dst_ckpt = os.path.join(out_dir, "checkpoint_tta.pt")
    torch.save(
        (pft3a.base_model1.state_dict(),
         pft3a.optimizer_list[0].state_dict()),
        dst_ckpt)
    print(f"[save] tta checkpoint -> {dst_ckpt}", flush=True)

    metrics_payload = {
        "experiment": experiment,
        "model": model,
        "source_ckpt": ckpt_path,
        "n_epochs": n_epochs,
        "batch_size": bs,
        "splits": {
            "ood_test": {
                "n": int(label_arr.shape[0]),
                "accuracy": float(acc),
                "balanced_accuracy": float(bal_acc),
                "f1": float(f1),
            },
        },
        "tta": {
            "total_used": int(total_used),
            "total_seen": int(total_seen),
            "fraction": float(total_used / max(total_seen, 1)),
        },
    }
    metrics_path = os.path.join(out_dir, "metrics_tta.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[save] tta metrics    -> {metrics_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default="/data/xli/skw/Advanced-ML/data/TableShift")
    parser.add_argument("--ckpt_root", default="./models",
                        help="Directory holding <experiment>/<arch>/checkpoint.pt")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--experiment", default="assistments",
                        choices=SUPPORTED_EXPERIMENTS)
    parser.add_argument("--model", default="mlp")
    parser.add_argument("--n_epochs", type=int, default=1,
                        help="TTA epochs over the OOD loader.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size "
                             "(default: 1024, or 256 for tabtransformer).")
    args = parser.parse_args()
    main(**vars(args))
    print(f"[exit] main returned at {time.strftime('%H:%M:%S')}, "
          f"forcing os._exit(0) to skip slow interpreter shutdown",
          flush=True)
    os._exit(0)
