from dataclasses import asdict, dataclass
import os

import numpy as np

from .gmr_config import GMRCorrectionConfig
from .gmr_diagnostics import (
    build_gmr_diagnostic_record,
    build_reference_from_prior,
    compute_mean_field_metrics,
    save_gmr_diagnostics,
)
from .gmr_regularizer import GMRRegularizer


def select_global_guard_blend(candidate_scores, before_score, policy="min_score", tolerance=0.0, min_geo_gain=0.03):
    finite_scores = [
        (float(b), float(s)) for b, s in candidate_scores.items() if np.isfinite(b) and np.isfinite(s)
    ]
    if not finite_scores or not np.isfinite(before_score):
        return 0.0, float(before_score), float(before_score)

    tolerance = max(float(tolerance), 0.0)
    min_geo_gain = float(np.clip(min_geo_gain, 0.0, 1.0))
    if policy == "minimal_gain":
        target_score = float(before_score) * (1.0 - min_geo_gain)
        for blend, score in sorted(finite_scores, key=lambda item: item[0]):
            if score <= target_score:
                return blend, score, target_score
        improving = [(b, s) for b, s in finite_scores if s < before_score]
        if improving:
            blend, score = min(improving, key=lambda item: item[1])
            return blend, score, target_score
        return 0.0, float(before_score), target_score

    accepted_score = float(before_score) * (1.0 + tolerance)
    blend, score = min(finite_scores, key=lambda item: item[1])
    if score > accepted_score:
        return 0.0, float(before_score), accepted_score
    return blend, score, accepted_score


def compute_production_guard_delta(previous_prod_nrmse, best_prod_nrmse, prior_prod_nrmse, tolerance=0.05):
    values = (previous_prod_nrmse, best_prod_nrmse, prior_prod_nrmse)
    if any(v is None or not np.isfinite(v) for v in values):
        return 0.0
    if prior_prod_nrmse <= 0.0:
        return 0.0
    rel_deg = (previous_prod_nrmse - best_prod_nrmse) / prior_prod_nrmse
    return float(max(0.0, rel_deg - max(tolerance, 0.0)))


@dataclass
class GMRDirectionConfig:
    enabled: bool = True
    vae_ckpt: str = os.path.join("gmr_refactor", "checkpoints", "gmr_vae.pt")
    flow_ckpt: str = os.path.join("gmr_refactor", "checkpoints", "gmr_flow.pt")
    dataset_meta: str = os.path.join("gmr_refactor", "data", "prior_patch_dataset_meta.json")
    device: str = "auto"
    batch_size: int = 256
    max_realizations: int | None = None
    activate_from_iter: int = 1
    lambda_base: float = 0.20
    lambda_min: float = 0.0
    lambda_max: float = 0.30
    score_step_base: float = 0.05
    score_clip: float = 3.0
    strength_mode: str = "log_squash"
    violation_ref: float = 5.0
    stage_mode: str = "increasing"
    stage_floor: float = 0.0
    prod_guard_beta: float = 4.0
    prod_guard_tolerance: float = 0.05
    force_lambda: float | None = None
    diagnostics_dir: str = ""
    method_name: str = "APBO_GMR"
    run_id: str = "gmr_direction"
    write_diagnostics: bool = True
    global_guard_enabled: bool = True
    global_guard_tolerance: float = 0.0
    global_guard_policy: str = "min_score"
    global_guard_min_geo_gain: float = 0.03

    def correction_config(self):
        return GMRCorrectionConfig(
            lambda_base=self.lambda_base,
            lambda_min=self.lambda_min,
            lambda_max=self.lambda_max,
            score_step_base=self.score_step_base,
            score_clip=self.score_clip,
            activate_from_iter=self.activate_from_iter,
            strength_mode=self.strength_mode,
            violation_ref=self.violation_ref,
            stage_mode=self.stage_mode,
            stage_floor=self.stage_floor,
            prod_guard_beta=self.prod_guard_beta,
            prod_guard_tolerance=self.prod_guard_tolerance,
        )


class GMRDirectionCorrector:
    def __init__(self, config: GMRDirectionConfig, nx, ny, nz, geology_reference=None):
        self.config = config
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.geology_reference = geology_reference
        self._regularizer = None

    @property
    def regularizer(self):
        if self._regularizer is None:
            self._regularizer = GMRRegularizer(
                self.config.vae_ckpt,
                self.config.flow_ckpt,
                self.config.dataset_meta,
                correction_config=self.config.correction_config(),
                device=self.config.device,
            )
        return self._regularizer

    @staticmethod
    def _geo_score(metrics):
        keys = (
            "hist_dist_logperm",
            "hist_dist_poro",
            "variogram_error_logperm",
            "variogram_error_poro",
        )
        total = sum(float(metrics[k]) for k in keys if metrics.get(k) is not None and np.isfinite(metrics[k]))
        conn = metrics.get("connectivity_delta_logperm")
        if conn is not None and np.isfinite(conn):
            total += abs(float(conn))
        return float(total)

    def _apply_global_structure_guard(self, before_perm, before_poro, corrected_perm, corrected_poro, gmr_diag):
        if not self.config.global_guard_enabled or self.geology_reference is None:
            gmr_diag["global_guard_active"] = False
            return corrected_perm, corrected_poro, gmr_diag

        before_geo = compute_mean_field_metrics(
            before_perm, before_poro, self.geology_reference, self.nx, self.ny, self.nz
        )
        before_score = self._geo_score(before_geo)
        candidate_scores = {}
        for blend in (1.0, 0.75, 0.50, 0.25, 0.0):
            cand_perm = before_perm + blend * (corrected_perm - before_perm)
            cand_poro = before_poro + blend * (corrected_poro - before_poro)
            metrics = compute_mean_field_metrics(
                cand_perm, cand_poro, self.geology_reference, self.nx, self.ny, self.nz
            )
            candidate_scores[blend] = self._geo_score(metrics)

        tolerance = max(self.config.global_guard_tolerance, 0.0)
        best_blend, best_score, target_score = select_global_guard_blend(
            candidate_scores,
            before_score,
            policy=self.config.global_guard_policy,
            tolerance=tolerance,
            min_geo_gain=self.config.global_guard_min_geo_gain,
        )
        best_perm = before_perm + best_blend * (corrected_perm - before_perm)
        best_poro = before_poro + best_blend * (corrected_poro - before_poro)

        raw_projection_norm = gmr_diag.get("projection_norm_relative_perm")
        selected_projection_norm = float(
            np.linalg.norm(best_perm - before_perm) / max(np.linalg.norm(before_perm), 1e-12)
        )
        gmr_diag.update({
            "global_guard_active": True,
            "global_guard_applied": best_blend < 1.0,
            "global_guard_blend": best_blend,
            "global_guard_score_before": before_score,
            "global_guard_score_after_raw": candidate_scores.get(1.0, np.nan),
            "global_guard_score_after_selected": best_score,
            "global_guard_candidate_scores": {str(k): v for k, v in candidate_scores.items()},
            "global_guard_tolerance": tolerance,
            "global_guard_policy": self.config.global_guard_policy,
            "global_guard_min_geo_gain": self.config.global_guard_min_geo_gain,
            "global_guard_target_score": target_score,
            "global_guard_geo_gain_selected": (before_score - best_score) / max(abs(before_score), 1e-12),
            "projection_norm_relative_perm_raw": raw_projection_norm,
            "projection_norm_relative_perm": selected_projection_norm,
            "global_guard_latent_diagnostics_are_raw": best_blend < 1.0,
        })
        return best_perm.astype(np.float32), best_poro.astype(np.float32), gmr_diag

    def set_reference_from_prior(self, prior_perm, prior_poro, effective=None):
        self.geology_reference = build_reference_from_prior(
            prior_perm, prior_poro, self.nx, self.ny, self.nz, effective=effective
        )

    def apply(
        self,
        mumyperm_after_esmda,
        mumyporo_after_esmda,
        mumypermz_after_esmda=None,
        iteration=1,
        total_iterations=4,
        prod_delta=None,
        prod_nrmse=None,
        prod_mismatch_legacy=None,
    ):
        if not self.config.enabled:
            return mumyperm_after_esmda, mumyporo_after_esmda, mumypermz_after_esmda, {"gmr_active": False, "reason": "disabled"}
        if iteration < self.config.activate_from_iter:
            return mumyperm_after_esmda, mumyporo_after_esmda, mumypermz_after_esmda, {"gmr_active": False, "reason": "before_activate_from_iter"}

        before_perm = np.asarray(mumyperm_after_esmda, dtype=np.float32).copy()
        before_poro = np.asarray(mumyporo_after_esmda, dtype=np.float32).copy()
        corrected_perm, corrected_poro, gmr_diag = self.regularizer.correct_ensemble(
            before_perm,
            before_poro,
            mumypermz=mumypermz_after_esmda,
            iteration=iteration,
            total_iterations=total_iterations,
            prod_delta=prod_delta,
            batch_size=self.config.batch_size,
            max_realizations=self.config.max_realizations,
            force_lambda=self.config.force_lambda,
        )
        corrected_perm, corrected_poro, gmr_diag = self._apply_global_structure_guard(
            before_perm, before_poro, corrected_perm, corrected_poro, gmr_diag
        )
        gmr_diag["gmr_active"] = True
        gmr_diag["integration_config"] = asdict(self.config)

        if self.config.write_diagnostics and self.config.diagnostics_dir and self.geology_reference is not None:
            csv_path = os.path.join(self.config.diagnostics_dir, "gmr_diagnostics.csv")
            json_path = os.path.join(self.config.diagnostics_dir, f"gmr_diagnostics_iter{iteration:02d}.json")
            record = build_gmr_diagnostic_record(
                method=self.config.method_name,
                run_id=self.config.run_id,
                iteration=iteration,
                before_perm=before_perm,
                before_poro=before_poro,
                after_perm=corrected_perm,
                after_poro=corrected_poro,
                gmr_diagnostics=gmr_diag,
                geology_reference=self.geology_reference,
                nx=self.nx,
                ny=self.ny,
                nz=self.nz,
                prod_nrmse=prod_nrmse,
                prod_mismatch_legacy=prod_mismatch_legacy,
            )
            save_gmr_diagnostics(csv_path, json_path, record, details={"gmr_diagnostics": gmr_diag})
            gmr_diag["diagnostics_csv"] = csv_path
            gmr_diag["diagnostics_json"] = json_path

        return corrected_perm, corrected_poro, mumypermz_after_esmda, gmr_diag


def make_gmr_direction_corrector(
    config: GMRDirectionConfig,
    nx,
    ny,
    nz,
    prior_perm=None,
    prior_poro=None,
    effective=None,
    geology_reference=None,
):
    corrector = GMRDirectionCorrector(config, nx=nx, ny=ny, nz=nz, geology_reference=geology_reference)
    if geology_reference is None and prior_perm is not None and prior_poro is not None:
        corrector.set_reference_from_prior(prior_perm, prior_poro, effective=effective)
    return corrector
