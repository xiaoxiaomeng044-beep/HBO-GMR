import argparse
import os
from dataclasses import asdict, dataclass

import numpy as np

from .common import save_json, vecN_to_xyzN


DEFAULT_CHANNELS = ("log10_perm_x", "poro")


@dataclass(frozen=True)
class PriorPatchDatasetConfig:
    nx: int = 19
    ny: int = 28
    nz: int = 5
    n_ensemble: int = 200
    patch_size_xyz: tuple = (4, 16, 4)
    n_patch_coords: int = 128
    min_valid_fraction: float = 0.75
    val_fraction: float = 0.15
    seed: int = 2025
    channels: tuple = DEFAULT_CHANNELS
    permx_file: str = "sgsim.out"
    poro_file: str = "sgsimporo.out"
    permz_file: str = "sgz.out"
    effective_file: str = "effective.out"


def _read_vec_file(path, expected_size):
    with open(path, "r", encoding="utf-8") as f:
        arr = np.fromiter(f, float)
    if arr.size != expected_size:
        raise ValueError(f"{path} has {arr.size} values; expected {expected_size}")
    return arr.astype(np.float32)


def _channel_array(name, fields):
    if name == "log10_perm_x":
        return np.log10(np.maximum(fields["perm_x"], 1e-15)).astype(np.float32)
    if name == "poro":
        return fields["poro"].astype(np.float32)
    if name == "log10_perm_z":
        return np.log10(np.maximum(fields["perm_z"], 1e-15)).astype(np.float32)
    raise ValueError(f"Unknown channel: {name}")


def load_prior_fields(config: PriorPatchDatasetConfig):
    expected = config.nx * config.ny * config.nz * config.n_ensemble
    fields = {
        "perm_x": vecN_to_xyzN(
            np.reshape(_read_vec_file(config.permx_file, expected), (config.nx * config.ny * config.nz, config.n_ensemble), order="F"),
            config.nx,
            config.ny,
            config.nz,
        ),
        "poro": vecN_to_xyzN(
            np.reshape(_read_vec_file(config.poro_file, expected), (config.nx * config.ny * config.nz, config.n_ensemble), order="F"),
            config.nx,
            config.ny,
            config.nz,
        ),
    }
    if "log10_perm_z" in config.channels:
        fields["perm_z"] = vecN_to_xyzN(
            np.reshape(_read_vec_file(config.permz_file, expected), (config.nx * config.ny * config.nz, config.n_ensemble), order="F"),
            config.nx,
            config.ny,
            config.nz,
        )
    effective = _read_vec_file(config.effective_file, config.nx * config.ny * config.nz)
    effective_xyz = np.reshape(effective.reshape(-1) != 0, (config.nx, config.ny, config.nz), order="F")
    return fields, effective_xyz


def enumerate_valid_patch_coords(effective_xyz, patch_size_xyz, min_valid_fraction):
    px, py, pz = patch_size_xyz
    nx, ny, nz = effective_xyz.shape
    if px > nx or py > ny or pz > nz:
        raise ValueError(f"patch_size_xyz={patch_size_xyz} exceeds grid shape {(nx, ny, nz)}")

    coords = []
    valid_fracs = []
    for ix in range(nx - px + 1):
        for iy in range(ny - py + 1):
            for iz in range(nz - pz + 1):
                frac = float(np.mean(effective_xyz[ix:ix + px, iy:iy + py, iz:iz + pz]))
                if frac >= min_valid_fraction:
                    coords.append((ix, iy, iz))
                    valid_fracs.append(frac)
    if not coords:
        raise ValueError(f"No valid patch coordinates found for min_valid_fraction={min_valid_fraction}")
    return np.asarray(coords, dtype=np.int32), np.asarray(valid_fracs, dtype=np.float32)


def select_fixed_patch_bank(coords, valid_fractions, n_patch_coords, seed):
    n_available = coords.shape[0]
    n_take = min(n_patch_coords, n_available)
    rng = np.random.default_rng(seed)
    if n_take >= n_available:
        selected = np.arange(n_available)
    else:
        weights = np.asarray(valid_fractions, dtype=np.float64)
        weights = weights / max(float(np.sum(weights)), 1e-12)
        selected = np.sort(rng.choice(n_available, size=n_take, replace=False, p=weights))
    return coords[selected].astype(np.int32), valid_fractions[selected].astype(np.float32)


def compute_channel_stats(channel_fields, effective_xyz, eps=1e-8):
    means = []
    stds = []
    mask = effective_xyz[..., None]
    for arr in channel_fields:
        vals = np.asarray(arr, dtype=np.float32)[mask.repeat(arr.shape[3], axis=3)]
        vals = vals[np.isfinite(vals)]
        means.append(float(np.mean(vals)))
        stds.append(float(max(np.std(vals), eps)))
    return np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)


def build_patch_tensor(channel_fields, coords_xyz, realization_ids, patch_size_xyz, channel_mean, channel_std):
    px, py, pz = patch_size_xyz
    mean = channel_mean.reshape(-1, 1, 1, 1)
    std = channel_std.reshape(-1, 1, 1, 1)
    samples = []
    sample_realizations = []
    sample_coord_indices = []
    for rid in realization_ids:
        for coord_idx, (ix, iy, iz) in enumerate(coords_xyz):
            patch = np.stack(
                [np.transpose(arr[ix:ix + px, iy:iy + py, iz:iz + pz, rid], (2, 1, 0)).astype(np.float32) for arr in channel_fields],
                axis=0,
            )
            samples.append(((patch - mean) / (std + 1e-8)).astype(np.float32))
            sample_realizations.append(int(rid))
            sample_coord_indices.append(coord_idx)
    return (
        np.stack(samples, axis=0).astype(np.float32),
        np.asarray(sample_realizations, dtype=np.int32),
        np.asarray(sample_coord_indices, dtype=np.int32),
    )


def split_realizations(n_ensemble, val_fraction, seed):
    rng = np.random.default_rng(seed)
    ids = np.arange(n_ensemble, dtype=np.int32)
    rng.shuffle(ids)
    n_val = max(1, int(round(val_fraction * n_ensemble)))
    return np.sort(ids[n_val:]).astype(np.int32), np.sort(ids[:n_val]).astype(np.int32)


def build_prior_patch_dataset(config: PriorPatchDatasetConfig, out_npz, out_meta):
    fields, effective_xyz = load_prior_fields(config)
    channel_fields = [_channel_array(name, fields) for name in config.channels]
    all_coords, all_valid_fracs = enumerate_valid_patch_coords(
        effective_xyz, config.patch_size_xyz, config.min_valid_fraction
    )
    coords_xyz, valid_fractions = select_fixed_patch_bank(
        all_coords, all_valid_fracs, config.n_patch_coords, config.seed
    )
    channel_mean, channel_std = compute_channel_stats(channel_fields, effective_xyz)
    train_ids, val_ids = split_realizations(config.n_ensemble, config.val_fraction, config.seed + 17)

    patches_train, train_realization_ids, train_coord_indices = build_patch_tensor(
        channel_fields, coords_xyz, train_ids, config.patch_size_xyz, channel_mean, channel_std
    )
    patches_val, val_realization_ids, val_coord_indices = build_patch_tensor(
        channel_fields, coords_xyz, val_ids, config.patch_size_xyz, channel_mean, channel_std
    )

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        patches_train=patches_train,
        patches_val=patches_val,
        patch_coords_xyz=coords_xyz.astype(np.int32),
        patch_valid_fraction=valid_fractions.astype(np.float32),
        train_realization_ids=train_realization_ids,
        val_realization_ids=val_realization_ids,
        train_coord_indices=train_coord_indices,
        val_coord_indices=val_coord_indices,
        train_realizations=train_ids,
        val_realizations=val_ids,
        channel_mean=channel_mean,
        channel_std=channel_std,
    )

    meta = {
        "config": asdict(config),
        "output_npz": out_npz,
        "patch_tensor_convention": "samples, channels, depth, height, width",
        "patch_coord_convention": "ix, iy, iz in PUNQ xyz grid",
        "patch_size_dhw": [config.patch_size_xyz[2], config.patch_size_xyz[1], config.patch_size_xyz[0]],
        "n_available_patch_coords": int(all_coords.shape[0]),
        "n_selected_patch_coords": int(coords_xyz.shape[0]),
        "n_train_realizations": int(train_ids.size),
        "n_val_realizations": int(val_ids.size),
        "n_train_samples": int(patches_train.shape[0]),
        "n_val_samples": int(patches_val.shape[0]),
        "patches_train_shape": list(patches_train.shape),
        "patches_val_shape": list(patches_val.shape),
        "channel_mean": channel_mean.astype(float).tolist(),
        "channel_std": channel_std.astype(float).tolist(),
        "source_files": {
            "permx": config.permx_file,
            "poro": config.poro_file,
            "permz": config.permz_file if "log10_perm_z" in config.channels else None,
            "effective": config.effective_file,
        },
    }
    save_json(meta, out_meta)
    return meta


def main():
    parser = argparse.ArgumentParser(description="Build GMR prior patch dataset from SGS prior ensemble.")
    parser.add_argument("--out_npz", default=os.path.join("gmr_refactor", "data", "prior_patch_dataset.npz"))
    parser.add_argument("--out_meta", default=os.path.join("gmr_refactor", "data", "prior_patch_dataset_meta.json"))
    parser.add_argument("--nx", type=int, default=19)
    parser.add_argument("--ny", type=int, default=28)
    parser.add_argument("--nz", type=int, default=5)
    parser.add_argument("--n_ensemble", type=int, default=200)
    parser.add_argument("--patch_size_xyz", default="4,16,4")
    parser.add_argument("--n_patch_coords", type=int, default=128)
    parser.add_argument("--min_valid_fraction", type=float, default=0.75)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--channels", default="log10_perm_x,poro")
    parser.add_argument("--permx_file", default="sgsim.out")
    parser.add_argument("--poro_file", default="sgsimporo.out")
    parser.add_argument("--permz_file", default="sgz.out")
    parser.add_argument("--effective_file", default="effective.out")
    args = parser.parse_args()

    patch_xyz = tuple(int(p.strip()) for p in args.patch_size_xyz.split(",") if p.strip())
    if len(patch_xyz) != 3:
        raise ValueError("patch_size_xyz must have 3 comma-separated integers")

    config = PriorPatchDatasetConfig(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        n_ensemble=args.n_ensemble,
        patch_size_xyz=patch_xyz,
        n_patch_coords=args.n_patch_coords,
        min_valid_fraction=args.min_valid_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
        channels=tuple(c.strip() for c in args.channels.split(",") if c.strip()),
        permx_file=args.permx_file,
        poro_file=args.poro_file,
        permz_file=args.permz_file,
        effective_file=args.effective_file,
    )
    meta = build_prior_patch_dataset(config, out_npz=args.out_npz, out_meta=args.out_meta)
    print(f"[GMR Dataset] Saved: {meta['output_npz']}")
    print(f"[GMR Dataset] train={meta['patches_train_shape']} val={meta['patches_val_shape']}")


if __name__ == "__main__":
    main()
