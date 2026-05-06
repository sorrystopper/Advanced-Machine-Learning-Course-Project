"""Lightweight CSV-backed substitute for TabularDataset.

Loads pre-split, pre-preprocessed CSVs produced for the TableShift benchmark
(layout: ``{data_dir}/{name}/{name}_X{split}.csv`` and ``..._y{split}.csv`` with
split in ``{train, val, idtest, ood}``) and exposes the subset of the
``TabularDataset`` interface that the PFT3A training / TTA pipelines actually
touch (see plan in repo notes).

Categoricals are inferred from the train split: any column whose values are a
subset of {0, 1} is treated as a one-hot dummy (the upstream preprocessor
emits these by default for these 6 datasets).
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


_CSV_TO_DSET_SPLIT = {
    "train": "train",
    "validation": "val",
    "id_test": "idtest",
    "ood_test": "ood",
}


def _make_loader(X: pd.DataFrame, y: pd.Series, G: pd.DataFrame,
                 batch_size: int, shuffle: bool) -> DataLoader:
    tensors = (
        torch.tensor(X.values).float(),
        torch.tensor(y.values).float(),
        torch.tensor(G.values).float(),
    )
    return DataLoader(TensorDataset(*tensors),
                      batch_size=batch_size, shuffle=shuffle)


class CSVTabularDataset:
    """Minimal dset that satisfies the PFT3A training pipeline's contract."""

    def __init__(self, name: str, data_dir: str):
        self.name = name
        self._dir = os.path.join(data_dir, name)
        if not os.path.isdir(self._dir):
            raise FileNotFoundError(f"dataset directory not found: {self._dir}")

        self._cache: dict = {}
        X_train, y_train = self._load("train")

        self._n_train = len(X_train)
        self._X_shape = X_train.shape

        is_binary = X_train.isin([0, 1]).all(axis=0)
        self._cat_idxs = [i for i, flag in enumerate(is_binary.values) if flag]

        p = float(y_train.mean())
        self._source_y_prior = (1.0 - p, p)

    def _csv_paths(self, dset_split: str) -> Tuple[str, str]:
        csv_split = _CSV_TO_DSET_SPLIT[dset_split]
        return (
            os.path.join(self._dir, f"{self.name}_X{csv_split}.csv"),
            os.path.join(self._dir, f"{self.name}_y{csv_split}.csv"),
        )

    def _load(self, dset_split: str) -> Tuple[pd.DataFrame, pd.Series]:
        if dset_split in self._cache:
            return self._cache[dset_split]
        x_path, y_path = self._csv_paths(dset_split)
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
        if y.shape[1] != 1:
            raise ValueError(
                f"expected single-column label CSV at {y_path}, got shape {y.shape}")
        y = y.iloc[:, 0]
        self._cache[dset_split] = (X, y)
        return X, y

    @property
    def is_domain_split(self) -> bool:
        return True

    @property
    def eval_split_names(self) -> Tuple[str, ...]:
        return ("validation", "id_test", "ood_test")

    @property
    def X_shape(self) -> Tuple[int, int]:
        return self._X_shape

    @property
    def cat_idxs(self):
        return list(self._cat_idxs)

    @property
    def n_train(self) -> int:
        return self._n_train

    @property
    def n_domains(self) -> int:
        return 1

    @property
    def source_y_prior(self) -> Tuple[float, float]:
        return self._source_y_prior

    def get_pandas(self, split: str, domain=None):
        if domain is not None:
            raise NotImplementedError("per-domain access not supported")
        X, y = self._load(split)
        G = pd.DataFrame(np.zeros((len(X), 1), dtype=np.float32),
                         columns=["_group"])
        return X, y, G, None

    def get_dataloader(self, split: str, batch_size: int = 2048,
                       shuffle: Optional[bool] = None,
                       infinite: bool = False) -> DataLoader:
        if infinite:
            raise NotImplementedError("infinite dataloader not supported")
        X, y, G, _ = self.get_pandas(split)
        if shuffle is None:
            shuffle = (split == "train")
        return _make_loader(X, y, G, batch_size=batch_size, shuffle=shuffle)

    def evaluate_predictions(self, preds, split: str) -> dict:
        _, y, _, _ = self.get_pandas(split)
        preds = np.asarray(preds).reshape(-1)
        if preds.dtype != np.int_ and not np.array_equal(preds, preds.astype(int)):
            preds = (preds >= 0.5).astype(int)
        acc = float((preds == y.values.astype(int)).mean())
        return {f"accuracy_{split}": acc}
