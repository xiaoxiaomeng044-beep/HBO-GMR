import json
import os

import numpy as np
import torch

from ae_model import AttentionAE3D


def _build_patch_bank(D, H, W, pd, ph, pw, n_patches=2000, seed=2025):
    z_max = max(1, D - pd + 1)
    y_max = max(1, H - ph + 1)
    x_max = max(1, W - pw + 1)
    total = z_max * y_max * x_max
    rng = np.random.default_rng(seed)

    if n_patches >= total:
        return np.array(
            [(iz, iy, ix) for iz in range(z_max) for iy in range(y_max) for ix in range(x_max)],
            dtype=np.int32,
        )

    flat_idx = rng.choice(total, size=n_patches, replace=False)
    coords = []
    for idx in flat_idx:
        iz = idx // (y_max * x_max)
        rem = idx % (y_max * x_max)
        coords.append((int(iz), int(rem // x_max), int(rem % x_max)))
    return np.array(coords, dtype=np.int32)


@torch.no_grad()
def compute_phi_ae_for_field(
    log10k,
    poro,
    ckpt_dir="ae_ckpt",
    n_patches=2000,
    agg="median",
    device="cpu",
    return_details=False,
    save_npz=None,
    topk=8,
    seed=2025,
    coords_xyz=None,
    coords_dhw=None,
):
    meta_path = os.path.join(ckpt_dir, "ae_meta.json")
    ckpt_path = os.path.join(ckpt_dir, "ae_kp.pt")
    if not (os.path.exists(meta_path) and os.path.exists(ckpt_path)):
        raise FileNotFoundError(f"Missing ae_meta.json or ae_kp.pt in {ckpt_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    patch_x, patch_y, patch_z = meta["patch"]
    ph, pw, pd = patch_y, patch_x, patch_z
    mean = np.array(meta["mean"], dtype=np.float32)
    std = np.array(meta["std"], dtype=np.float32)

    x0 = (log10k.astype(np.float32) - mean[0]) / (std[0] + 1e-8)
    x1 = (poro.astype(np.float32) - mean[1]) / (std[1] + 1e-8)
    field = np.stack([x0, x1], axis=0)[None, ...]
    _, _, D, H, W = field.shape

    model = AttentionAE3D(in_ch=2, base_ch=32, z_down=True).to(device).eval()
    state = torch.load(ckpt_path, map_location="cpu")
    state = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(state, strict=False)

    if coords_dhw is not None:
        coords = np.asarray(coords_dhw, dtype=np.int32)
    elif coords_xyz is not None:
        coords_xyz = np.asarray(coords_xyz, dtype=np.int32)
        coords = np.stack([coords_xyz[:, 2], coords_xyz[:, 1], coords_xyz[:, 0]], axis=1).astype(np.int32)
    else:
        coords = _build_patch_bank(D, H, W, pd, ph, pw, n_patches=n_patches, seed=seed)

    errs = []
    fea_mean = []
    sum_map = np.zeros((D, H, W), dtype=np.float32)
    cnt_map = np.zeros((D, H, W), dtype=np.int32)
    top_inputs, top_recons, top_coords, top_errs = [], [], [], []

    for iz, iy, ix in coords:
        patch = field[:, :, iz:iz + pd, iy:iy + ph, ix:ix + pw]
        xb = torch.from_numpy(patch).to(device)
        yb = model(xb)
        mae = (yb - xb).abs().mean().item()

        errs.append(mae)
        fea_mean.append((float(xb[0, 0].mean().item()), float(xb[0, 1].mean().item())))
        sum_map[iz:iz + pd, iy:iy + ph, ix:ix + pw] += mae
        cnt_map[iz:iz + pd, iy:iy + ph, ix:ix + pw] += 1

        if topk > 0 and (len(top_errs) < topk or mae > min(top_errs)):
            xb_np = xb.cpu().numpy()[0]
            yb_np = yb.cpu().numpy()[0]
            xb_np[0] = xb_np[0] * (std[0] + 1e-8) + mean[0]
            xb_np[1] = xb_np[1] * (std[1] + 1e-8) + mean[1]
            yb_np[0] = yb_np[0] * (std[0] + 1e-8) + mean[0]
            yb_np[1] = yb_np[1] * (std[1] + 1e-8) + mean[1]

            top_inputs.append(xb_np.copy())
            top_recons.append(yb_np.copy())
            top_coords.append((int(iz), int(iy), int(ix)))
            top_errs.append(mae)

            if len(top_errs) > topk:
                j = int(np.argmin(top_errs))
                for lst in (top_inputs, top_recons, top_coords, top_errs):
                    lst.pop(j)

    errs = np.array(errs, dtype=np.float32)
    fea_mean = np.array(fea_mean, dtype=np.float32)
    avg_map = sum_map / np.maximum(cnt_map, 1)
    phi_raw = float(np.quantile(errs, 0.9) if agg.lower() == "q90" else np.median(errs))

    details = None
    if return_details or save_npz:
        details = {
            "phi_raw": phi_raw,
            "errs": errs,
            "coords": coords,
            "fea_mean": fea_mean,
            "q50": float(np.quantile(errs, 0.5)),
            "q90": float(np.quantile(errs, 0.9)),
            "q95": float(np.quantile(errs, 0.95)),
            "mean": float(errs.mean()),
            "std": float(errs.std()),
            "patch_shape": (int(pd), int(ph), int(pw)),
            "avg_error_map": avg_map.astype(np.float32),
            "cnt_map": cnt_map.astype(np.int32),
        }
        if topk > 0 and top_inputs:
            order = np.argsort(top_errs)[::-1]
            details.update({
                "topk_inputs": np.stack([top_inputs[i] for i in order], axis=0),
                "topk_recons": np.stack([top_recons[i] for i in order], axis=0),
                "topk_coords": np.stack([top_coords[i] for i in order], axis=0),
                "topk_errs": np.array([top_errs[i] for i in order], dtype=np.float32),
            })

    if save_npz:
        os.makedirs(os.path.dirname(save_npz), exist_ok=True)
        np.savez_compressed(
            save_npz,
            phi_raw=phi_raw,
            errs=errs,
            coords=coords,
            fea_mean=fea_mean,
            q50=np.quantile(errs, 0.5),
            q90=np.quantile(errs, 0.9),
            q95=np.quantile(errs, 0.95),
            mean=errs.mean(),
            std=errs.std(),
            patch_D=int(pd),
            patch_H=int(ph),
            patch_W=int(pw),
            avg_error_map=avg_map.astype(np.float32),
            cnt_map=cnt_map.astype(np.int32),
            topk_inputs=details.get("topk_inputs") if details else None,
            topk_recons=details.get("topk_recons") if details else None,
            topk_coords=details.get("topk_coords") if details else None,
            topk_errs=details.get("topk_errs") if details else None,
        )

    return (phi_raw, details) if return_details else phi_raw
