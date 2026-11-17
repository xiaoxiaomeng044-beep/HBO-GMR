from dataclasses import dataclass


@dataclass(frozen=True)
class GMRVAEConfig:
    channels: tuple = ("log10_perm_x", "poro")
    patch_size: tuple = (4, 16, 4)
    latent_dim: int = 32
    beta_kl: float = 1.0
    attention: str = "channel_spatial_3d"


@dataclass(frozen=True)
class GMRFlowConfig:
    flow_type: str = "realnvp"
    hidden_dim: int = 128
    num_layers: int = 6


@dataclass(frozen=True)
class GMRCorrectionConfig:
    nll_q_low: float = 0.75
    nll_q_high: float = 0.90
    lambda_base: float = 0.20
    lambda_min: float = 0.0
    lambda_max: float = 0.30
    score_step_base: float = 0.05
    score_clip: float = 3.0
    activate_from_iter: int = 1
    strength_mode: str = "log_squash"
    violation_ref: float = 5.0
    stage_mode: str = "increasing"
    stage_floor: float = 0.0
    prod_guard_beta: float = 4.0
    prod_guard_tolerance: float = 0.05
