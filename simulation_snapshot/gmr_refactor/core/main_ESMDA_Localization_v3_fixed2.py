import numpy as np

from ESMDA_Localization2_v3_fixed2 import ESMDA_Localization2
from honour2 import honour2


def main_ESMDA_Localization(
    nx, ny, nz, N,
    observation, overallsim,
    rossmary, rossmaryporo, rossmaryz2,
    mumyperm, mumyporo, mumypermz,
    history, alpha, effective,
    c_radius=5.0,
    use_localization=True,
    update_status=None,
    ae_pseudo=None,
    ae_time_window=3,
    ae_time_sigma_multipliers=None,
    ae_update_weight=1.0,
    n_obs=19,
    n_producers=6,
    well_locations=None,
    param_bounds=None,
    random_seed=None,
    observation_error_member_indices=None,
    observation_error_bank_size=None,
):
    effective = effective.flatten()
    mumyperm = mumyperm * effective[:, None]
    mumyporo = mumyporo * effective[:, None]
    mumypermz = mumypermz * effective[:, None]

    indices = np.where(effective)[0]
    mumyperm2 = mumyperm[indices, :]
    mumyporo2 = mumyporo[indices, :]
    mumypermz2 = mumypermz[indices, :]

    ae_time_window = max(1, min(history, ae_time_window or 1))
    if ae_time_sigma_multipliers is None:
        ae_time_sigma_multipliers = np.linspace(1.35, 1.0, ae_time_window).tolist() if ae_time_window > 1 else [1.0]
    else:
        ae_time_sigma_multipliers = list(ae_time_sigma_multipliers)
        if len(ae_time_sigma_multipliers) < ae_time_window:
            tail = ae_time_sigma_multipliers[-1] if ae_time_sigma_multipliers else 1.0
            ae_time_sigma_multipliers.extend([tail] * (ae_time_window - len(ae_time_sigma_multipliers)))
        ae_time_sigma_multipliers = ae_time_sigma_multipliers[:ae_time_window]

    for i in range(history):
        if update_status:
            update_status(i + 1, history)

        sim1 = overallsim[i, :, :].reshape(n_obs, N)
        f = observation[:, i]

        ae_pseudo_step = None
        if ae_pseudo is not None and i >= history - ae_time_window:
            local_idx = i - (history - ae_time_window)
            ae_pseudo_step = dict(ae_pseudo)
            if ae_pseudo_step.get("sigma_AE") is not None:
                ae_pseudo_step["sigma_AE"] = float(ae_pseudo_step["sigma_AE"]) * float(ae_time_sigma_multipliers[local_idx])
            ae_pseudo_step["time_window_local_index"] = local_idx
            ae_pseudo_step["time_window_sigma_mult"] = float(ae_time_sigma_multipliers[local_idx])

        mumyperm2, mumyporo2, mumypermz2 = ESMDA_Localization2(
            mumyperm2, mumyporo2, mumypermz2,
            f, N, sim1, alpha, indices, c_radius,
            use_localization=use_localization,
            ae_pseudo=ae_pseudo_step,
            nx=nx, ny=ny, nz=nz,
            well_locations=well_locations,
            n_obs=n_obs,
            n_producers=n_producers,
            ae_update_weight=ae_update_weight,
            random_seed=None if random_seed is None else random_seed + i,
            observation_error_member_indices=observation_error_member_indices,
            observation_error_bank_size=observation_error_bank_size,
        )

    if param_bounds is None:
        param_bounds = {
            "permx": (0.5073, 999.1116),
            "poro": (0.01024, 0.2992),
            "permz": (0.1478, 498.4011),
        }

    DupdateK = np.clip(np.abs(mumyperm2), *param_bounds["permx"])
    sgsim2 = np.clip(np.abs(mumyporo2), *param_bounds["poro"])
    sgz1 = np.clip(np.abs(mumypermz2), *param_bounds["permz"])

    DupdateK2 = np.ones((nx * ny * nz, N))
    sgsim2f = np.ones((nx * ny * nz, N))
    sgz1f = np.ones((nx * ny * nz, N))
    DupdateK2[indices, :] = DupdateK
    sgsim2f[indices, :] = sgsim2
    sgz1f[indices, :] = sgz1

    return honour2(
        rossmary, rossmaryporo, rossmaryz2, N,
        sgz1f, sgsim2f, DupdateK2,
        nx=nx, ny=ny, nz=nz, well_locations=well_locations,
    )
