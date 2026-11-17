import numpy as np


def compute_j_unified(prod_score, phi_ae, prod0, phi0, lambda_phi=1.0, eps=1e-12):
    if prod0 is None or phi0 is None or prod0 <= 0 or phi0 <= 0:
        return np.nan
    if not np.isfinite(prod_score) or not np.isfinite(phi_ae):
        return np.nan
    return float((prod_score / (prod0 + eps)) + lambda_phi * (phi_ae / (phi0 + eps)))


def compute_j_bo(j_unified, alpha, lambda_budget=0.03, eps=1e-12):
    if not np.isfinite(j_unified):
        return np.nan
    return float(j_unified + lambda_budget * (1.0 / max(float(alpha), eps)))
