import csv
import hashlib
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_csv_row(path, row, encoding="utf-8-sig"):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def sha256_json(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def vecN_to_xyzN(vecN, nx, ny, nz):
    mat = np.asarray(vecN)
    _, n_ens = mat.shape
    arr_TN = np.transpose(mat, (1, 0))
    models = np.array(
        [np.reshape(arr_TN[i, :], (nx, ny, nz), order="F") for i in range(n_ens)],
        dtype=np.float32,
    )
    return np.transpose(models, (1, 2, 3, 0))


def xyzN_to_vecN(arr_xyzN):
    nx, ny, nz, n_ens = arr_xyzN.shape
    out = np.zeros((nx * ny * nz, n_ens), dtype=np.float32)
    for j in range(n_ens):
        out[:, j] = np.reshape(arr_xyzN[:, :, :, j], nx * ny * nz, order="F")
    return out


class NPZPatchDataset(Dataset):
    def __init__(self, npz_path, split="train"):
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self._npz = np.load(npz_path)
        self.patches = self._npz[f"patches_{split}"]

    def __len__(self):
        return int(self.patches.shape[0])

    def __getitem__(self, index):
        return torch.from_numpy(self.patches[index].astype(np.float32, copy=False))
