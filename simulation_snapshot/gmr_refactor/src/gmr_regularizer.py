from dataclasses import asdict
import os

import numpy as np
import torch

from .attention_vae_3d import AttentionVAE3D
from .common import load_json, vecN_to_xyzN, xyzN_to_vecN
from .gmr_config import GMRCorrectionConfig
from .latent_flow import RealNVPFlow


def _channel_fields_from_vecN(mumyperm, mumyporo, nx, ny, nz, channels, mumypermz=None):
    fields = {
        "log10_perm_x": np.log10(np.maximum(vecN_to_xyzN(mumyperm, nx, ny, nz), 1e-15)).astype(np.float32),
        "poro": vecN_to_xyzN(mumyporo, nx, ny, nz).astype(np.float32),
    }
    if "log10_perm_z" in channels:
        if mumypermz is None:
            raise ValueError("Channel log10_perm_z requested but mumypermz is None")
        fields["log10_perm_z"] = np.log10(np.maximum(vecN_to_xyzN(mumypermz, nx, ny, nz), 1e-15)).astype(np.float32)
    return fields


def _extract_patch_dhw(channel_fields, channels, coord_xyz, realization_id, patch_size_xyz, channel_mean, channel_std):
    ix, iy, iz = coord_xyz
    px, py, pz = patch_size_xyz
    patches = [
        np.transpose(channel_fields[name][ix:ix + px, iy:iy + py, iz:iz + pz, realization_id], (2, 1, 0)).astype(np.float32)
        for name in channels
    ]
    patch = np.stack(patches, axis=0)
    return ((patch - channel_mean.reshape(-1, 1, 1, 1)) / (channel_std.reshape(-1, 1, 1, 1) + 1e-8)).astype(np.float32)


def _denormalize_patch_dhw(patch, channel_mean, channel_std):
    return patch * channel_std.reshape(-1, 1, 1, 1) + channel_mean.reshape(-1, 1, 1, 1)


def _safe_quantile(values, q):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, q)) if arr.size else float("nan")


class GMRRegularizer:
    def __init__(self, vae_ckpt, flow_ckpt, dataset_meta, correction_config=None, device="auto"):
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dataset_meta_path = dataset_meta
        self.dataset_meta = load_json(dataset_meta)
        self.config = correction_config or GMRCorrectionConfig()
        self.channels = tuple(self.dataset_meta["config"]["channels"])
        self.nx = int(self.dataset_meta["config"]["nx"])
        self.ny = int(self.dataset_meta["config"]["ny"])
        self.nz = int(self.dataset_meta["config"]["nz"])
        self.patch_size_xyz = tuple(int(v) for v in self.dataset_meta["config"]["patch_size_xyz"])
        self.patch_coords_xyz = self._load_patch_coords_from_dataset_meta()
        self.channel_mean = np.asarray(self.dataset_meta["channel_mean"], dtype=np.float32)
        self.channel_std = np.asarray(self.dataset_meta["channel_std"], dtype=np.float32)

        self.vae, self.vae_checkpoint = self._load_vae(vae_ckpt)
        self.flow, self.flow_checkpoint = self._load_flow(flow_ckpt)
        self.prior_nll_q50, self.prior_nll_q75, self.prior_nll_q90 = self._load_prior_nll_quantiles()

    def _load_patch_coords_from_dataset_meta(self):
        npz_path = self.dataset_meta.get("output_npz")
        if not npz_path or not os.path.exists(npz_path):
            local_fallback = os.path.join(os.path.dirname(os.path.abspath(self.dataset_meta_path)), "prior_patch_dataset.npz")
            if os.path.exists(local_fallback):
                npz_path = local_fallback
            else:
                npz_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "prior_patch_dataset.npz")
        with np.load(npz_path) as z:
            return z["patch_coords_xyz"].astype(np.int32)

    def _load_vae(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg = ckpt["model_config"]
        model = AttentionVAE3D(
            in_channels=int(cfg["in_channels"]),
            input_shape_dhw=tuple(int(v) for v in cfg["input_shape_dhw"]),
            latent_dim=int(cfg["latent_dim"]),
            base_channels=int(cfg["base_channels"]),
        ).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model, ckpt

    def _load_flow(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg = ckpt["flow_config"]
        model = RealNVPFlow(
            dim=int(cfg["dim"]),
            hidden_dim=int(cfg["hidden_dim"]),
            num_layers=int(cfg["num_layers"]),
            scale_clip=float(cfg.get("scale_clip", 2.0)),
        ).to(self.device)
        model.load_state_dict(ckpt["flow_state_dict"], strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model, ckpt

    def _load_prior_nll_quantiles(self):
        q50 = float(self.flow_checkpoint.get("best_val_nll_q50", np.nan))
        q75 = float(self.flow_checkpoint.get("best_val_nll_q75", np.nan))
        q90 = float(self.flow_checkpoint.get("best_val_nll_q90", np.nan))
        if not np.isfinite(q75) or not np.isfinite(q90) or q90 <= q75:
            q50 = float(self.flow_checkpoint.get("best_val_nll", 0.0))
            q75 = q50
            q90 = q50 + 1.0
        return q50, q75, q90

    def build_patch_batch(self, mumyperm, mumyporo, mumypermz=None, max_realizations=None):
        fields = _channel_fields_from_vecN(
            mumyperm, mumyporo, self.nx, self.ny, self.nz, self.channels, mumypermz=mumypermz
        )
        n_ens = mumyperm.shape[1] if max_realizations is None else min(mumyperm.shape[1], max_realizations)
        patches, realization_ids, coord_indices = [], [], []
        for j in range(n_ens):
            for coord_idx, coord in enumerate(self.patch_coords_xyz):
                patches.append(
                    _extract_patch_dhw(
                        fields, self.channels, coord, j,
                        self.patch_size_xyz, self.channel_mean, self.channel_std,
                    )
                )
                realization_ids.append(j)
                coord_indices.append(coord_idx)
        return (
            np.stack(patches, axis=0).astype(np.float32),
            np.asarray(realization_ids, dtype=np.int32),
            np.asarray(coord_indices, dtype=np.int32),
        )

    @torch.no_grad()
    def compute_latent_diagnostics(self, mumyperm, mumyporo, mumypermz=None, batch_size=256, max_realizations=None):
        patches, _, _ = self.build_patch_batch(
            mumyperm, mumyporo, mumypermz=mumypermz, max_realizations=max_realizations
        )
        nll_values = []
        for start in range(0, patches.shape[0], batch_size):
            xb = torch.from_numpy(patches[start:start + batch_size]).to(self.device)
            z = self.vae.encode_mu(xb)
            nll_values.append(self.flow.nll(z).detach().cpu().numpy())
        nll_all = np.concatenate(nll_values) if nll_values else np.asarray([], dtype=float)
        return {
            "patch_count": int(patches.shape[0]),
            "latent_nll_mean": float(np.mean(nll_all)) if nll_all.size else None,
            "latent_nll_q50": _safe_quantile(nll_all, 0.50),
            "latent_nll_q75": _safe_quantile(nll_all, 0.75),
            "latent_nll_q90": _safe_quantile(nll_all, 0.90),
            "prior_nll_q50": self.prior_nll_q50,
            "prior_nll_q75": self.prior_nll_q75,
            "prior_nll_q90": self.prior_nll_q90,
            "violation_ratio": self.compute_violation_ratio(nll_all),
        }

    def compute_violation_ratio(self, nll_values):
        current_q75 = _safe_quantile(nll_values, 0.75)
        denom = max(self.prior_nll_q90 - self.prior_nll_q75, 1e-8)
        return float(max(0.0, (current_q75 - self.prior_nll_q75) / denom))

    def compute_adaptive_strength(self, violation_ratio, iteration, total_iterations, prod_delta=None):
        cfg = self.config
        denom = max(total_iterations - cfg.activate_from_iter, 1)
        stage_floor = float(np.clip(cfg.stage_floor, 0.0, 1.0))
        if cfg.stage_mode == "late_taper":
            progress = np.clip((iteration - cfg.activate_from_iter) / denom, 0.0, 1.0)
            stage = max(stage_floor, 1.0 - float(progress))
        else:
            stage = min(1.0, max(0.0, (iteration - cfg.activate_from_iter + 1) / denom))

        prod_guard_beta = max(cfg.prod_guard_beta, 0.0)
        prod_guard_tolerance = max(cfg.prod_guard_tolerance, 0.0)
        prod_guard_delta = max(float(prod_delta), 0.0) if prod_delta is not None and np.isfinite(prod_delta) else 0.0
        fit_guard = float(np.exp(-prod_guard_beta * prod_guard_delta)) if prod_guard_delta > 0.0 else 1.0

        if cfg.strength_mode == "log_squash":
            v = float(np.log1p(max(violation_ratio, 0.0)))
            v_norm = v / (v + max(cfg.violation_ref, 1e-8))
            lambda_before_prod_guard = cfg.lambda_max * v_norm * stage
        else:
            v_norm = float(violation_ratio)
            lambda_before_prod_guard = cfg.lambda_base * violation_ratio * stage

        raw_lambda = lambda_before_prod_guard * fit_guard
        lambda_gmr = float(np.clip(raw_lambda, cfg.lambda_min, cfg.lambda_max))
        eta_score = cfg.score_step_base * stage
        return lambda_gmr, eta_score, {
            "stage_mode": cfg.stage_mode,
            "stage_factor": stage,
            "stage_floor": stage_floor,
            "prod_guard_delta_norm": prod_guard_delta,
            "prod_guard_beta": prod_guard_beta,
            "prod_guard_tolerance": prod_guard_tolerance,
            "fit_guard": fit_guard,
            "lambda_before_prod_guard": lambda_before_prod_guard,
            "raw_lambda": raw_lambda,
            "strength_mode": cfg.strength_mode,
            "violation_squashed": v_norm,
        }

    def correct_ensemble(
        self,
        mumyperm,
        mumyporo,
        mumypermz=None,
        iteration=1,
        total_iterations=4,
        prod_delta=None,
        batch_size=256,
        max_realizations=None,
        force_lambda=None,
    ):
        n_ens_full = mumyperm.shape[1]
        n_ens = n_ens_full if max_realizations is None else min(n_ens_full, max_realizations)
        patches, realization_ids, coord_indices = self.build_patch_batch(
            mumyperm, mumyporo, mumypermz=mumypermz, max_realizations=n_ens
        )

        nll_values = []
        with torch.no_grad():
            for start in range(0, patches.shape[0], batch_size):
                xb = torch.from_numpy(patches[start:start + batch_size]).to(self.device)
                nll_values.append(self.flow.nll(self.vae.encode_mu(xb)).detach().cpu().numpy())

        nll_all = np.concatenate(nll_values) if nll_values else np.asarray([], dtype=float)
        violation = self.compute_violation_ratio(nll_all)
        lambda_gmr, eta_score, strength_diag = self.compute_adaptive_strength(
            violation, iteration=iteration, total_iterations=total_iterations, prod_delta=prod_delta
        )
        if force_lambda is not None:
            lambda_gmr = float(force_lambda)

        decoded_norm = np.zeros_like(patches, dtype=np.float32)
        score_norms = []
        for start in range(0, patches.shape[0], batch_size):
            xb = torch.from_numpy(patches[start:start + batch_size]).to(self.device)
            with torch.no_grad():
                z = self.vae.encode_mu(xb)
            score = self.flow.score(z)
            score_norm = torch.linalg.norm(score, dim=1, keepdim=True).clamp_min(1e-8)
            score_unit = torch.clamp(score / score_norm, -self.config.score_clip, self.config.score_clip)
            with torch.no_grad():
                recon = self.vae.decode(z + eta_score * score_unit, output_shape_dhw=xb.shape[2:])
            decoded_norm[start:start + batch_size] = recon.detach().cpu().numpy().astype(np.float32)
            score_norms.append(score_norm.detach().cpu().numpy().reshape(-1))

        corrected_perm_xyzN = vecN_to_xyzN(mumyperm, self.nx, self.ny, self.nz).astype(np.float32)
        corrected_poro_xyzN = vecN_to_xyzN(mumyporo, self.nx, self.ny, self.nz).astype(np.float32)
        accum_perm = np.zeros_like(corrected_perm_xyzN[:, :, :, :n_ens], dtype=np.float32)
        accum_poro = np.zeros_like(corrected_poro_xyzN[:, :, :, :n_ens], dtype=np.float32)
        weights = np.zeros((self.nx, self.ny, self.nz, n_ens), dtype=np.float32)

        decoded_original = _denormalize_patch_dhw(decoded_norm, self.channel_mean, self.channel_std)
        px, py, pz = self.patch_size_xyz
        for sample_idx, (rid, coord_idx) in enumerate(zip(realization_ids, coord_indices)):
            ix, iy, iz = self.patch_coords_xyz[coord_idx]
            patch = decoded_original[sample_idx]
            for ch_idx, name in enumerate(self.channels):
                patch_xyz = np.transpose(patch[ch_idx], (2, 1, 0)).astype(np.float32)
                if name == "log10_perm_x":
                    accum_perm[ix:ix + px, iy:iy + py, iz:iz + pz, rid] += np.power(10.0, patch_xyz)
                elif name == "poro":
                    accum_poro[ix:ix + px, iy:iy + py, iz:iz + pz, rid] += patch_xyz
            weights[ix:ix + px, iy:iy + py, iz:iz + pz, rid] += 1.0

        covered = weights > 0
        projected_perm = corrected_perm_xyzN[:, :, :, :n_ens].copy()
        projected_poro = corrected_poro_xyzN[:, :, :, :n_ens].copy()
        projected_perm[covered] = accum_perm[covered] / weights[covered]
        projected_poro[covered] = accum_poro[covered] / weights[covered]

        corrected_perm_xyzN[:, :, :, :n_ens] += lambda_gmr * (projected_perm - corrected_perm_xyzN[:, :, :, :n_ens])
        corrected_poro_xyzN[:, :, :, :n_ens] += lambda_gmr * (projected_poro - corrected_poro_xyzN[:, :, :, :n_ens])
        corrected_perm_xyzN = np.maximum(corrected_perm_xyzN, 1e-15)
        corrected_poro_xyzN = np.clip(corrected_poro_xyzN, 1e-6, 0.60)

        perm_out = xyzN_to_vecN(corrected_perm_xyzN).astype(np.float32)
        poro_out = xyzN_to_vecN(corrected_poro_xyzN).astype(np.float32)
        projection_norm = float(
            np.linalg.norm(perm_out[:, :n_ens] - mumyperm[:, :n_ens])
            / max(np.linalg.norm(mumyperm[:, :n_ens]), 1e-12)
        )
        after_latent = self.compute_latent_diagnostics(
            perm_out[:, :n_ens],
            poro_out[:, :n_ens],
            mumypermz=mumypermz[:, :n_ens] if mumypermz is not None else None,
            batch_size=batch_size,
            max_realizations=n_ens,
        )
        diagnostics = {
            "patch_count": int(patches.shape[0]),
            "realization_count": int(n_ens),
            "latent_nll_mean": float(np.mean(nll_all)) if nll_all.size else None,
            "latent_nll_q50": _safe_quantile(nll_all, 0.50),
            "latent_nll_q75": _safe_quantile(nll_all, 0.75),
            "latent_nll_q90": _safe_quantile(nll_all, 0.90),
            "prior_nll_q50": self.prior_nll_q50,
            "prior_nll_q75": self.prior_nll_q75,
            "prior_nll_q90": self.prior_nll_q90,
            "violation_ratio": violation,
            "lambda_gmr": lambda_gmr,
            "eta_score": eta_score,
            "projection_norm_relative_perm": projection_norm,
            "latent_nll_after_mean": after_latent.get("latent_nll_mean"),
            "latent_nll_after_q50": after_latent.get("latent_nll_q50"),
            "latent_nll_after_q75": after_latent.get("latent_nll_q75"),
            "latent_nll_after_q90": after_latent.get("latent_nll_q90"),
            "delta_latent_nll_mean": (
                float(after_latent["latent_nll_mean"]) - float(np.mean(nll_all))
                if after_latent.get("latent_nll_mean") is not None and nll_all.size
                else None
            ),
            "delta_latent_nll_q75": (
                float(after_latent["latent_nll_q75"]) - _safe_quantile(nll_all, 0.75)
                if after_latent.get("latent_nll_q75") is not None
                else None
            ),
            "score_norm_mean": float(np.mean(np.concatenate(score_norms))) if score_norms else None,
            **strength_diag,
            "config": asdict(self.config),
        }
        return perm_out, poro_out, diagnostics
