import json
import os

import numpy as np
import torch

from ae_model import AttentionAE3D


def try_load_ae(ckpt_dir="ae_ckpt"):
    ckpt = os.path.join(ckpt_dir, "ae_kp.pt")
    meta = os.path.join(ckpt_dir, "ae_meta.json")
    if not (os.path.exists(ckpt) and os.path.exists(meta)):
        return None

    with open(meta, "r", encoding="utf-8") as f:
        m = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionAE3D(in_ch=2, base_ch=32, z_down=True).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    stats = {
        "mean": np.array(m["mean"], dtype=np.float32).reshape(1, 2, 1, 1, 1),
        "std": np.array(m["std"], dtype=np.float32).reshape(1, 2, 1, 1, 1),
    }
    patch = tuple(m.get("patch", (4, 16, 16)))
    return {"device": device, "model": model, "stats": stats, "patch": patch}


def _reshape_vecN_to_xyzN(vecN, nx, ny, nz):
    arr_TN = np.transpose(vecN, (1, 0))
    models = np.array(
        [np.reshape(arr_TN[i, :], (nx, ny, nz), order="F") for i in range(arr_TN.shape[0])],
        dtype=np.float32,
    )
    return np.transpose(models, (1, 2, 3, 0))


def _pad_to_depth(vol, need):
    if vol.shape[2] >= need:
        return vol
    pad = need - vol.shape[2]
    left = pad // 2
    return np.pad(vol, ((0, 0), (0, 0), (left, pad - left)), mode="reflect")


def _patch_recon_mae(batch_np, model, stats, device):
    xn = (batch_np - stats["mean"]) / (stats["std"] + 1e-8)
    xb = torch.from_numpy(xn).to(device)
    with torch.no_grad():
        yb = model(xb)
        err = torch.mean(torch.abs(yb - xb), dim=(1, 2, 3, 4)).cpu().numpy()
    return err.astype(np.float32)


def _patch_recon(batch_np, model, stats, device):
    xn = (batch_np - stats["mean"]) / (stats["std"] + 1e-8)
    xb = torch.from_numpy(xn).to(device)
    with torch.no_grad():
        yb = model(xb)
        err = torch.mean(torch.abs(yb - xb), dim=(1, 2, 3, 4)).cpu().numpy()
    recon = yb.cpu().numpy() * (stats["std"] + 1e-8) + stats["mean"]
    return recon.astype(np.float32), err.astype(np.float32)


def _sample_candidate_coords(nx, ny, nz, px, py, z_span, n_candidates, rng):
    return [
        (
            int(rng.integers(0, max(1, nx - px + 1))),
            int(rng.integers(0, max(1, ny - py + 1))),
            int(rng.integers(0, max(1, nz - z_span + 1))),
            z_span,
        )
        for _ in range(n_candidates)
    ]


def build_patch_bank_xyz(nx, ny, nz, px, py, pz, n_patches=128, seed=2025):
    x_max = max(1, nx - px + 1)
    y_max = max(1, ny - py + 1)
    z_max = max(1, nz - pz + 1)
    total = x_max * y_max * z_max

    if n_patches >= total:
        return np.array(
            [(ix, iy, iz) for ix in range(x_max) for iy in range(y_max) for iz in range(z_max)],
            dtype=np.int32,
        )

    rng = np.random.default_rng(seed)
    flat_idx = rng.choice(total, size=n_patches, replace=False)
    coords = []
    for idx in flat_idx:
        ix = idx // (y_max * z_max)
        rem = idx % (y_max * z_max)
        coords.append((int(ix), int(rem // z_max), int(rem % z_max)))
    return np.array(coords, dtype=np.int32)


def _select_diverse_hard_patches(candidate_errors, coords, n_patches, px, py, hard_ratio=0.6):
    n_candidates = len(coords)
    n_hard = max(1, int(round(n_patches * hard_ratio)))
    n_rand = max(0, n_patches - n_hard)

    mean_err = np.mean(candidate_errors, axis=1)
    cand_xyz = np.array([(c[0], c[1], c[2]) for c in coords], dtype=np.float32)
    sorted_idx = np.argsort(mean_err)[::-1]

    selected = []
    selected_centers = []
    min_dist_sq = (0.7 * px) ** 2 + (0.7 * py) ** 2

    for idx in sorted_idx:
        if len(selected) >= n_hard:
            break
        cc = cand_xyz[idx]
        if all(np.sum((cc - sc) ** 2) >= min_dist_sq for sc in selected_centers):
            selected.append(idx)
            selected_centers.append(cc)

    if len(selected) < n_hard:
        remaining = [i for i in sorted_idx if i not in selected]
        selected.extend(remaining[: n_hard - len(selected)])

    remaining = [i for i in range(n_candidates) if i not in selected]
    if n_rand > 0 and remaining:
        rng_local = np.random.default_rng(12345)
        selected.extend(rng_local.choice(remaining, size=min(n_rand, len(remaining)), replace=False).tolist())

    return np.array(selected[:n_patches], dtype=int)


def build_ae_pseudo(
    mumyperm,
    mumyporo,
    nx,
    ny,
    nz,
    ae_bundle,
    n_patches=128,
    seed=2025,
    pad_to_trained=True,
    sigma_AE=None,
    coords_xyz=None,
    target_quantile=0.15,
    hard_ratio=0.35,
):
    if ae_bundle is None:
        return None

    rng = np.random.default_rng(seed)
    device = ae_bundle["device"]
    model = ae_bundle["model"]
    stats = ae_bundle["stats"]
    px_tr, py_tr, pz_tr = ae_bundle["patch"]

    K_xyzN = _reshape_vecN_to_xyzN(mumyperm, nx, ny, nz)
    P_xyzN = _reshape_vecN_to_xyzN(mumyporo, nx, ny, nz)

    px = min(px_tr, nx)
    py = min(py_tr, ny)
    pz = pz_tr if pad_to_trained else min(pz_tr, nz)
    z_span = min(pz, nz)

    if coords_xyz is not None:
        coords_xyz = np.asarray(coords_xyz, dtype=np.int32)[:n_patches]
        coords = [(int(ix), int(iy), int(iz), z_span) for ix, iy, iz in coords_xyz]
    else:
        coords = _sample_candidate_coords(nx, ny, nz, px, py, z_span, max(n_patches * 4, n_patches), rng)

    n_ens = K_xyzN.shape[3]
    n_candidates = len(coords)
    candidate_errors = np.zeros((n_candidates, n_ens), dtype=np.float32)

    for m, (ix, iy, iz, zz) in enumerate(coords):
        patches = []
        for j in range(n_ens):
            k = K_xyzN[ix:ix + px, iy:iy + py, iz:iz + zz, j]
            p = P_xyzN[ix:ix + px, iy:iy + py, iz:iz + zz, j]
            if pad_to_trained and k.shape[2] < pz:
                k = _pad_to_depth(k, pz)
                p = _pad_to_depth(p, pz)
            patches.append(np.stack([np.log10(np.maximum(k, 1e-15)).astype(np.float32), p.astype(np.float32)], axis=0))
        candidate_errors[m, :] = _patch_recon_mae(np.stack(patches, axis=0), model, stats, device)

    if coords_xyz is not None:
        selected_idx = np.arange(n_candidates, dtype=int)
    else:
        selected_idx = _select_diverse_hard_patches(candidate_errors, coords, n_patches, px, py, hard_ratio=hard_ratio)

    D_AE_final = candidate_errors[selected_idx, :]
    q = float(np.clip(target_quantile, 0.0, 0.49))
    y_AE = np.quantile(D_AE_final, q, axis=1).astype(np.float32)

    if sigma_AE is None:
        flat = (D_AE_final - y_AE[:, None]).ravel()
        med = np.median(flat)
        mad = np.median(np.abs(flat - med)) + 1e-12
        sigma_AE = float(max(1.4826 * mad, np.std(flat) + 1e-12, 1e-4)) * 1.25

    selected_coords_xyz = np.array([(coords[i][0], coords[i][1], coords[i][2]) for i in selected_idx], dtype=np.int32)
    selected_centers_xyz = np.array(
        [
            (
                coords[i][0] + 0.5 * max(px - 1, 0),
                coords[i][1] + 0.5 * max(py - 1, 0),
                coords[i][2] + 0.5 * max(coords[i][3] - 1, 0),
            )
            for i in selected_idx
        ],
        dtype=np.float32,
    )

    return {
        "D_AE": D_AE_final.astype(np.float32),
        "y_AE": y_AE,
        "sigma_AE": float(sigma_AE),
        "coords_xyz": selected_coords_xyz,
        "centers_xyz": selected_centers_xyz,
        "patch_shape_xyz": np.array([px, py, z_span], dtype=np.int32),
        "grid_shape_xyz": np.array([nx, ny, nz], dtype=np.int32),
    }


def build_ae_recon_project_pseudo(
    mumyperm,
    mumyporo,
    nx,
    ny,
    nz,
    ae_bundle,
    n_patches=128,
    seed=2025,
    pad_to_trained=True,
    coords_xyz=None,
    effective=None,
    hard_ratio=0.30,
    max_obs=512,
    blend=1.0,
    active_fraction_min=0.75,
):
    if ae_bundle is None:
        return None

    rng = np.random.default_rng(seed)
    device = ae_bundle["device"]
    model = ae_bundle["model"]
    stats = ae_bundle["stats"]
    px_tr, py_tr, pz_tr = ae_bundle["patch"]

    K_raw = _reshape_vecN_to_xyzN(mumyperm, nx, ny, nz)
    P_raw = _reshape_vecN_to_xyzN(mumyporo, nx, ny, nz)

    active_mask = (
        np.ones((nx, ny, nz), dtype=bool)
        if effective is None
        else np.reshape(np.asarray(effective).reshape(-1) != 0, (nx, ny, nz), order="F")
    )

    active_k = K_raw[active_mask, :].ravel()
    active_k = active_k[np.isfinite(active_k) & (active_k > 1e-12)]
    active_p = P_raw[active_mask, :].ravel()
    active_p = active_p[np.isfinite(active_p)]
    if active_k.size == 0 or active_p.size == 0:
        return None

    logk_active = np.log10(np.maximum(active_k, 1e-12))
    logk_lo, logk_hi = np.percentile(logk_active, [1.0, 99.0])
    poro_lo, poro_hi = np.percentile(active_p, [1.0, 99.0])

    logK_xyzN = np.full_like(K_raw, float(np.median(logk_active)), dtype=np.float32)
    P_xyzN = np.full_like(P_raw, float(np.median(active_p)), dtype=np.float32)
    logK_xyzN[active_mask, :] = np.clip(np.log10(np.maximum(K_raw[active_mask, :], 1e-12)), logk_lo, logk_hi).astype(np.float32)
    P_xyzN[active_mask, :] = np.clip(P_raw[active_mask, :], poro_lo, poro_hi).astype(np.float32)

    logK_mean = np.mean(logK_xyzN, axis=3).astype(np.float32)
    poro_mean = np.mean(P_xyzN, axis=3).astype(np.float32)

    px = min(px_tr, nx)
    py = min(py_tr, ny)
    pz = pz_tr if pad_to_trained else min(pz_tr, nz)
    z_span = min(pz, nz)

    if coords_xyz is not None:
        coords_xyz = np.asarray(coords_xyz, dtype=np.int32)[:n_patches]
        coords = [(int(ix), int(iy), int(iz), z_span) for ix, iy, iz in coords_xyz]
    else:
        coords = _sample_candidate_coords(nx, ny, nz, px, py, z_span, max(n_patches * 4, n_patches), rng)

    patches, valid_coords, valid_masks = [], [], []
    for ix, iy, iz, zz in coords:
        active_crop = active_mask[ix:ix + px, iy:iy + py, iz:iz + zz]
        if active_crop.size == 0 or float(np.mean(active_crop)) < active_fraction_min:
            continue
        k = logK_mean[ix:ix + px, iy:iy + py, iz:iz + zz]
        p = poro_mean[ix:ix + px, iy:iy + py, iz:iz + zz]
        if pad_to_trained and k.shape[2] < pz:
            k = _pad_to_depth(k, pz)
            p = _pad_to_depth(p, pz)
            active_crop_pad = _pad_to_depth(active_crop.astype(np.float32), pz) > 0.5
        else:
            active_crop_pad = active_crop
        patches.append(np.stack([k, p], axis=0).astype(np.float32))
        valid_coords.append((ix, iy, iz, zz))
        valid_masks.append(active_crop_pad.astype(bool))

    if not patches:
        return None

    batch = np.stack(patches, axis=0)
    recon, errs = _patch_recon(batch, model, stats, device)
    n_hard = max(1, int(round(len(valid_coords) * np.clip(hard_ratio, 0.0, 1.0))))
    selected = np.argsort(errs)[::-1][:n_hard]

    target_logk = np.zeros((nx, ny, nz), dtype=np.float64)
    target_poro = np.zeros((nx, ny, nz), dtype=np.float64)
    score = np.zeros((nx, ny, nz), dtype=np.float64)
    weight = np.zeros((nx, ny, nz), dtype=np.float64)

    mean0, mean1 = float(stats["mean"].reshape(-1)[0]), float(stats["mean"].reshape(-1)[1])
    std0, std1 = float(stats["std"].reshape(-1)[0]), float(stats["std"].reshape(-1)[1])

    for idx in selected:
        ix, iy, iz, zz = valid_coords[int(idx)]
        cur = batch[int(idx), :, :, :, :zz]
        reb = recon[int(idx), :, :, :, :zz]
        act = valid_masks[int(idx)][:, :, :zz]
        delta_norm = np.where(
            act,
            np.abs((reb[0] - cur[0]) / (std0 + 1e-8)) + np.abs((reb[1] - cur[1]) / (std1 + 1e-8)),
            0.0,
        )
        slx, sly, slz = slice(ix, ix + px), slice(iy, iy + py), slice(iz, iz + zz)
        w = act.astype(np.float64) * float(errs[int(idx)])
        target_logk[slx, sly, slz] += np.clip(reb[0], logk_lo, logk_hi) * w
        target_poro[slx, sly, slz] += np.clip(reb[1], poro_lo, poro_hi) * w
        score[slx, sly, slz] += delta_norm * float(errs[int(idx)])
        weight[slx, sly, slz] += w

    mask = (weight > 0) & active_mask
    if not np.any(mask):
        return None

    target_logk[mask] /= np.maximum(weight[mask], 1e-12)
    target_poro[mask] /= np.maximum(weight[mask], 1e-12)

    max_cells = max(1, max(2, int(max_obs)) // 2)
    flat_candidates = np.flatnonzero(mask.ravel(order="F"))
    order = np.argsort(score.ravel(order="F")[flat_candidates])[::-1][:max_cells]
    flat_selected = flat_candidates[order]

    rows, y, row_centers, state_indices, state_kinds = [], [], [], [], []
    blend = float(np.clip(blend, 0.0, 1.0))

    for flat in flat_selected:
        ix = int(flat % nx)
        iy = int((flat // nx) % ny)
        iz = int(flat // (nx * ny))
        tgt_logk = np.clip((1.0 - blend) * logK_mean[ix, iy, iz] + blend * target_logk[ix, iy, iz], logk_lo, logk_hi)
        tgt_poro = np.clip((1.0 - blend) * poro_mean[ix, iy, iz] + blend * target_poro[ix, iy, iz], poro_lo, poro_hi)
        vec_idx = ix + nx * iy + nx * ny * iz

        rows.append((logK_xyzN[ix, iy, iz, :] - mean0) / (std0 + 1e-8))
        y.append((tgt_logk - mean0) / (std0 + 1e-8))
        row_centers.append((float(ix), float(iy), float(iz)))
        state_indices.append(vec_idx)
        state_kinds.append(0)

        rows.append((P_xyzN[ix, iy, iz, :] - mean1) / (std1 + 1e-8))
        y.append((tgt_poro - mean1) / (std1 + 1e-8))
        row_centers.append((float(ix), float(iy), float(iz)))
        state_indices.append(vec_idx)
        state_kinds.append(1)

    D = np.vstack(rows).astype(np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    flat_res = (D - y_arr[:, None]).ravel()
    flat_res = flat_res[np.isfinite(flat_res)]
    if flat_res.size == 0:
        sigma_AE = 1e-4
    else:
        med = np.median(flat_res)
        mad = np.median(np.abs(flat_res - med)) + 1e-12
        sigma_AE = float(max(1.4826 * mad, np.std(flat_res) + 1e-12, 1e-4) * 1.25)

    selected_coords = np.asarray(row_centers, dtype=np.float32)
    return {
        "D_AE": D,
        "y_AE": y_arr,
        "sigma_AE": float(sigma_AE),
        "coords_xyz": selected_coords.astype(np.int32),
        "centers_xyz": selected_coords,
        "patch_shape_xyz": np.array([px, py, z_span], dtype=np.int32),
        "grid_shape_xyz": np.array([nx, ny, nz], dtype=np.int32),
        "constraint_form": "recon_project",
        "target_mode": "ae_reconstruction_projection",
        "selected_patch_errors": errs[selected].astype(np.float32),
        "state_indices": np.asarray(state_indices, dtype=np.int32),
        "state_kinds": np.asarray(state_kinds, dtype=np.int32),
        "max_obs": max_obs,
        "blend": blend,
        "mean_patch_error": float(np.mean(errs)),
        "selected_mean_patch_error": float(np.mean(errs[selected])),
        "active_fraction_min": active_fraction_min,
        "logk_clip": np.array([logk_lo, logk_hi], dtype=np.float32),
        "poro_clip": np.array([poro_lo, poro_hi], dtype=np.float32),
        "active_cells": int(np.sum(active_mask)),
    }
