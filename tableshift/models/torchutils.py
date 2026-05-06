from typing import Union, Dict, Tuple

import numpy as np
import rtdl
import scipy
import copy
import sklearn
import torch
from tab_transformer_pytorch import TabTransformer
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from torch.nn import functional as F
from typing import Optional

from tableshift.third_party.saint.models import SAINT


def get_module_attr(model, attr):
    """Get an attribute from (possibly-distributed) module."""
    if hasattr(model, "module"):
        # Case: it is a distributed module; first access model attr explicitly.
        return getattr(model.module, attr)
    else:
        # Case: standard module; fetch the attr.
        return getattr(model, attr)


def unpack_batch(batch: Union[Dict, Tuple[Union[torch.Tensor, None]]]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Union[torch.Tensor, None]
]:
    if isinstance(batch, dict):
        # Case: dict-formatted batch; these are used for Ray training.
        x_batch = batch["x"]
        y_batch = batch["y"]
        g_batch = batch["g"]
        d_batch = batch.get("d", None)

    else:
        # Case: tuple of Tensors; these are used for vanilla Pytorch training.
        (x_batch, y_batch, g_batch) = batch[:3]
        d_batch = batch[3] if len(batch) == 4 else None

    return x_batch, y_batch, g_batch, d_batch


def split_num_cat(x, cat_idxs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Use cat_idxs to slice the input batch."""
    if not cat_idxs:
        x_cont = x
        x_categ = None
    else:
        assert max(cat_idxs) < x.shape[1], \
            f"expected cat_idxs in range [0, {x.shape[1] - 1}], got {cat_idxs}"
        categ_mask = np.isin(np.arange(x.shape[1]), cat_idxs)
        x_cont = x[:, ~categ_mask]
        x_categ = x[:, categ_mask]
    return x_cont, x_categ


def apply_model(model: torch.nn.Module, x):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        module = model.module
    else:
        module = model

    if isinstance(module, rtdl.FTTransformer) or isinstance(module, SAINT) \
            or isinstance(module, TabTransformer):
        x_num, x_cat = split_num_cat(x, get_module_attr(module, "cat_idxs"))
        return module(x_num, x_cat)

    else:
        return module(x)

def apply_model_hidden(model: torch.nn.Module, x):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        module = model.module
    else:
        module = model

    if isinstance(module, rtdl.FTTransformer) or isinstance(module, SAINT) \
            or isinstance(module, TabTransformer):
        x_num, x_cat = split_num_cat(x, get_module_attr(module, "cat_idxs"))
        return module.forward_hidden(x_num, x_cat)

    if isinstance(module, rtdl.MLP):
        return module.blocks(x)

    else:
        return module(x)


@torch.no_grad()
def get_predictions_and_labels(model, loader, device=None, as_logits=False) -> Tuple[
    np.ndarray, np.ndarray]:
    """Get the predictions (as logits, or probabilities) and labels."""
    prediction = []
    label = []

    if not device:
        device = f"cuda:{torch.cuda.current_device()}" \
        if torch.cuda.is_available() else "cpu"

    modelname = model.__class__.__name__
    for batch in tqdm(loader, desc=f"{modelname}:getpreds"):
        batch_x, batch_y, _, _ = unpack_batch(batch)
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        out = apply_model(model, batch_x)
        outputs = out[0] if isinstance(out, tuple) else out
        prediction.append(outputs)
        label.append(batch_y)
    prediction = torch.cat(prediction).squeeze().cpu().numpy()
    target = torch.cat(label).squeeze().cpu().numpy()
    if not as_logits:
        prediction = scipy.special.expit(prediction)
    return prediction, target

import tableshift.PFT3A_src.PFT3A as PFT3A

@torch.enable_grad()
def get_predictions_and_labels_tta_pft3a(model, loader, device, exp):
    optimizer = torch.optim.SGD
    prediction = []
    label = []
    target = []
    if exp == 'anes':
        source_y = torch.tensor([1 - 0.691, 0.691]).to(device)
    elif exp == 'heloc':
        source_y = torch.tensor([1 - 0.24, 0.24]).to(device)
    elif exp == 'assistments':
        source_y = torch.tensor([1 - 0.694, 0.694]).to(device)
    elif exp == 'brfss_blood_pressure':
        source_y = torch.tensor([1- 0.403, 0.403]).to(device)
    elif exp == 'brfss_diabetes':
        source_y = torch.tensor([1- 0.126, 0.126]).to(device)
    elif exp == 'acspubcov':
        source_y = torch.tensor([1- 0.224, 0.224]).to(device)
    elif hasattr(model, '_csv_source_y_prior'):
        source_y = torch.tensor(list(model._csv_source_y_prior)).to(device)
    else:
        print('please check exp name')
        raise ValueError("exp wrong")
    modelname = model.__class__.__name__
    source_y = torch.tensor([1- 0.5, 0.5]).to(device)

    pft3a = PFT3A.PFT3A(model, optimizer, prior = source_y, lr_list=[1e-5,5e-4,1e-4], device=device)
    
    #get source_y_estimation
    batch_x, _, _, _ = unpack_batch(next(iter(tqdm(loader, desc=f"{modelname}:getpreds"))))
    batch_x = batch_x.float().to(device)
    pft3a.get_prior(batch_x)
    
    for batch in tqdm(loader, desc=f"{modelname}:getpreds"):
        batch_x, batch_y, _, _ = unpack_batch(batch)
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        outputs = pft3a(batch_x, batch_y)
        prediction.append(outputs)
        label.append(batch_y)
    prediction = torch.cat(prediction).squeeze().cpu().numpy()
    target = torch.cat(label).squeeze().cpu().numpy()

    unique, counts = np.unique(target, return_counts=True)

    ratios = counts / len(target)

    for u, c, r in zip(unique, counts, ratios):
        print(f"Class {u}: {c} instances, {r:.2f} proportion")
    return prediction, target


import logging
def evaluate_tta(model, loader, device, split, exp:Optional[str] = None):
    if split == 'train':
        logging.info(f'TTA testing, only ood score will be display, split:{split} skipping')
        return 0
    if split == 'ood_test':
        with torch.enable_grad():
            model_ense = copy.deepcopy(model)
            model_ense.train()
            pre, tar = get_predictions_and_labels_tta_pft3a(model_ense, loader, device, exp)
            pre = np.round(pre)
            score = sklearn.metrics.accuracy_score(tar, pre)
            balanced_acc = sklearn.metrics.balanced_accuracy_score(tar, pre)
            f1 = sklearn.metrics.f1_score(tar, pre)
            print('\n', 'pft3a: acc ', '\n', score, '\n', 'pft3a: balanced_acc ', '\n', balanced_acc, '\n', 'pft3a: f1', '\n', f1, '\n')
        with torch.no_grad():
            model.eval()
            pre, tar = get_predictions_and_labels(model, loader, device)
            pre = np.round(pre)
            score = sklearn.metrics.accuracy_score(tar, pre)
            balanced_acc = sklearn.metrics.balanced_accuracy_score(tar, pre)
            f1 = sklearn.metrics.f1_score(tar, pre)
            print('\n', 'Unadapt: acc ', '\n', score, '\n', 'Unadapt: balanced_acc ', '\n', balanced_acc, '\n', 'Unadapt: f1', '\n', f1, '\n')
    else:
        logging.info(f'TTA testing, only ood score will be display, split:{split} skipping')
        return 0
    return score

def evaluate(model, loader, device, split, exp:Optional[str] = None):
    with torch.no_grad():
        model.eval()
        prediction, target = get_predictions_and_labels(model, loader, device)
        prediction = np.round(prediction)
        score = sklearn.metrics.accuracy_score(target, prediction)
        balanced = sklearn.metrics.balanced_accuracy_score(target, prediction)
        f1 = sklearn.metrics.f1_score(target, prediction, zero_division=0)
    print(f"[eval] split={split} n={len(target)} acc={score:.4f} "
          f"balanced_acc={balanced:.4f} f1={f1:.4f}", flush=True)
    if not hasattr(model, "_last_eval_metrics"):
        model._last_eval_metrics = {}
    model._last_eval_metrics[split] = {
        "n": int(len(target)),
        "accuracy": float(score),
        "balanced_accuracy": float(balanced),
        "f1": float(f1),
    }
    return score
