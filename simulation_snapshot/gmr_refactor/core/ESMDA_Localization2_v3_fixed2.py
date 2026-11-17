import numpy as np
from scipy.ndimage import distance_transform_edt


def _gaspari_cohn(r):
    r = np.asarray(r, dtype=np.float64)
    out = np.zeros_like(r)
    m1 = (r >= 0.0) & (r <= 1.0)
    m2 = (r > 1.0) & (r <= 2.0)
    if np.any(m1):
        x = r[m1]
        out[m1] = 1.0 - (5.0 / 3.0) * x**2 + (5.0 / 8.0) * x**3 + 0.5 * x**4 - 0.25 * x**5
    if np.any(m2):
        x = r[m2]
        out[m2] = 4.0 - 5.0 * x + (5.0 / 3.0) * x**2 + (5.0 / 8.0) * x**3 - 0.5 * x**4 + (1.0 / 12.0) * x**5
    return out


def _solve_with_jitter(A, B, jitters=(0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)):
    A = 0.5 * (np.asarray(A, dtype=np.float64) + np.asarray(A, dtype=np.float64).T)
    A = np.nan_to_num(A, nan=0.0, posinf=1e12, neginf=-1e12)
    B = np.nan_to_num(np.asarray(B, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    eye = np.eye(A.shape[0])
    for j in jitters:
        try:
            return np.linalg.solve(A + j * eye if j > 0.0 else A, B)
        except np.linalg.LinAlgError:
            continue
    return np.linalg.pinv(A, rcond=1e-8) @ B


def _flat_indices_to_xyz(indices, nx, ny, nz):
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    return np.stack([idx % nx, (idx // nx) % ny, idx // (nx * ny)], axis=1).astype(np.float64)


def _derive_ae_loc_radius(ae_pseudo, fallback=4.5):
    if ae_pseudo is None:
        return float(fallback)
    val = ae_pseudo.get("localization_radius")
    if val is not None:
        return max(float(val), 1e-6)
    patch_shape = np.asarray(ae_pseudo.get("patch_shape_xyz", [4, 8, 3]), dtype=np.float64).reshape(-1)
    if patch_shape.size >= 3:
        diag = float(np.sqrt(np.sum(np.square(patch_shape[:3]))))
        return max(3.5, min(6.5, 0.38 * diag + 0.8))
    return float(fallback)


def _build_ae_obs_localization(indices, ae_pseudo, nx, ny, nz, n_prod, n_static):
    if ae_pseudo is None:
        return None
    centers = ae_pseudo.get("centers_xyz")
    if centers is None or ae_pseudo.get("D_AE") is None:
        return None
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] == 0:
        return None

    active_xyz = _flat_indices_to_xyz(indices, nx, ny, nz)
    c_ae = _derive_ae_loc_radius(ae_pseudo, fallback=4.5)
    dist = np.sqrt(
        (active_xyz[:, None, 0] - centers[None, :, 0]) ** 2
        + (active_xyz[:, None, 1] - centers[None, :, 1]) ** 2
        + (active_xyz[:, None, 2] - centers[None, :, 2]) ** 2
    )
    W = _gaspari_cohn(dist / max(c_ae, 1e-12))
    M_AE = centers.shape[0]
    n_active = active_xyz.shape[0]
    W_state_obs = np.ones((n_static + n_prod + M_AE, M_AE), dtype=np.float64)
    W_state_obs[0:n_active, :] = W
    W_state_obs[n_active:2 * n_active, :] = W
    W_state_obs[2 * n_active:3 * n_active, :] = W
    return W_state_obs


def draw_observation_error_matrix(
    f,
    *,
    N,
    n_obs,
    n_producers,
    random_seed=None,
    member_indices=None,
    bank_size=None,
):
    f = np.asarray(f, dtype=np.float64).reshape(n_obs)
    if member_indices is None:
        ids = np.arange(N, dtype=int)
        draw_size = N
    else:
        ids = np.asarray(member_indices, dtype=int).reshape(-1)
        if ids.size != N:
            raise ValueError("member_indices length must equal N")
        draw_size = int(bank_size if bank_size is not None else int(ids.max()) + 1)
        if draw_size < N or np.any(ids < 0) or np.any(ids >= draw_size):
            raise ValueError("member_indices are outside the observation-error bank")

    rng = np.random.default_rng(random_seed)
    bank = np.zeros((n_obs, draw_size), dtype=np.float64)
    bank[0, :] = rng.normal(0.0, 0.1 * max(1e-12, f[0]), size=draw_size)
    for k in range(n_producers):
        bank[1 + k, :] = rng.normal(0.0, 0.15 * max(1e-12, f[1 + k]), size=draw_size)
    for k in range(n_producers):
        bank[1 + n_producers + k, :] = rng.normal(0.0, 0.15 * max(1e-12, f[1 + n_producers + k]), size=draw_size)
    for k in range(n_producers):
        bank[1 + 2 * n_producers + k, :] = rng.normal(0.0, 0.2 * max(1e-12, f[1 + 2 * n_producers + k]), size=draw_size)
    return bank[:, ids]


def ESMDA_Localization2(
    mumyperm2, mumyporo2, mumypermz2,
    f, N, Sim1, alpha, indices, c,
    use_localization=True,
    ae_pseudo=None,
    nx=19, ny=28, nz=5,
    well_locations=None,
    n_obs=19,
    n_producers=6,
    ae_update_weight=1.0,
    random_seed=None,
    observation_error_member_indices=None,
    observation_error_bank_size=None,
):
    n_active = mumyperm2.shape[0]
    total_cells = nx * ny * nz
    cells_per_layer = nx * ny

    if well_locations is None:
        well_locations = [(9, 21), (8, 16), (16, 10), (10, 23), (14, 16), (16, 21)]

    if use_localization and c is not None and np.isfinite(c) and float(c) > 0:
        well_grid = np.zeros((nx, ny, nz))
        for j in range(nz):
            for iw, jw in well_locations:
                if 0 <= iw < nx and 0 <= jw < ny:
                    well_grid[iw, jw, j] = 1
        lf = np.reshape(well_grid, (nx, ny, nz), "F")
        dist_layers = np.zeros((cells_per_layer, nz))
        for j in range(nz):
            dist_layers[:, j] = distance_transform_edt(lf[:, :, j]).reshape(cells_per_layer, order="F")
        dist = np.abs(dist_layers.reshape(total_cells, 1, order="F"))
        schuruse = np.asarray(_gaspari_cohn(dist / max(float(c), 1e-12))[indices, 0], dtype=np.float64)
    else:
        schuruse = np.ones(n_active, dtype=np.float64)
    Bsch = np.repeat(schuruse[:, None], N, axis=1)

    eps = 1e-10
    mumyperm2 = np.where(mumyperm2 <= 0, eps, mumyperm2)
    mumypermz2 = np.where(mumypermz2 <= 0, eps, mumypermz2)
    sgsim11 = np.log(mumyperm2).reshape(n_active, N, order="F")
    sgsim22 = np.log(mumypermz2).reshape(n_active, N, order="F")
    sgsim11poro = np.reshape(mumyporo2, (n_active, N), "F")

    f = np.asarray(f, dtype=np.float64).reshape(n_obs)
    Error1 = draw_observation_error_matrix(
        f,
        N=N,
        n_obs=n_obs,
        n_producers=n_producers,
        random_seed=random_seed,
        member_indices=observation_error_member_indices,
        bank_size=observation_error_bank_size,
    )
    rng = np.random.default_rng(random_seed)
    draw_size = int(observation_error_bank_size) if observation_error_member_indices is not None else N
    rng.normal(size=(n_obs, draw_size))

    Djuse = f[:, None] + Error1
    Simuse = np.asarray(Sim1[:n_obs, :], dtype=np.float64)
    Cd2 = (Error1 @ Error1.T) / max(N - 1, 1)

    n_static = 3 * n_active
    n_prod = n_obs
    use_ae = ae_pseudo is not None and "D_AE" in ae_pseudo

    if use_ae:
        D_AE = np.asarray(ae_pseudo["D_AE"], dtype=np.float64)
        y_AE = np.asarray(ae_pseudo.get("y_AE", 0.0), dtype=np.float64).reshape(-1)
        sigma_AE = float(ae_pseudo.get("sigma_AE", 0.1))
        M_AE = D_AE.shape[0]
        Dj_AE = y_AE.reshape(M_AE, 1) + rng.normal(0.0, sigma_AE, size=(M_AE, N))
        Dj_all = np.vstack([Djuse, Dj_AE])
        Sim_all = np.vstack([Simuse, D_AE])
        Cd_all = np.zeros((n_prod + M_AE, n_prod + M_AE), dtype=np.float64)
        Cd_all[:n_prod, :n_prod] = Cd2
        Cd_all[n_prod:, n_prod:] = (sigma_AE ** 2) * np.eye(M_AE, dtype=np.float64)
    else:
        M_AE = 0
        Dj_all = Djuse
        Sim_all = Simuse
        Cd_all = Cd2

    overall = np.zeros((n_static + n_prod + M_AE, N), dtype=np.float64)
    overall[0:n_active, :] = sgsim11
    overall[n_active:2 * n_active, :] = sgsim22
    overall[2 * n_active:3 * n_active, :] = sgsim11poro
    overall[3 * n_active:3 * n_active + n_prod, :] = Simuse
    if use_ae:
        overall[3 * n_active + n_prod:, :] = D_AE

    schur_prod = np.ones((n_static + n_prod + M_AE, N), dtype=np.float64)
    schur_prod[0:n_active, :] = Bsch
    schur_prod[n_active:2 * n_active, :] = Bsch
    schur_prod[2 * n_active:3 * n_active, :] = Bsch

    S = Sim_all - np.mean(Sim_all, axis=1, keepdims=True)
    yprime = overall - np.mean(overall, axis=1, keepdims=True)
    Cyd = (yprime @ S.T) / max(N - 1, 1)
    Cdd = (S @ S.T) / max(N - 1, 1)
    A = np.nan_to_num(Cdd + float(alpha) * Cd_all, nan=0.0, posinf=1e12, neginf=-1e12)
    gainR = _solve_with_jitter(A, Dj_all - Sim_all)

    if use_ae:
        update_prod = schur_prod * (Cyd[:, :n_prod] @ gainR[:n_prod, :])
        W_state_obs = (
            _build_ae_obs_localization(indices, ae_pseudo, nx, ny, nz, n_prod=n_prod, n_static=n_static)
            if use_localization
            else None
        )
        Cyd_ae = Cyd[:, n_prod:]
        gainR_ae = gainR[n_prod:, :]
        update_ae = (Cyd_ae * W_state_obs) @ gainR_ae if W_state_obs is not None else (Cyd_ae @ gainR_ae)
        Ynew = overall + update_prod + float(ae_update_weight) * update_ae
    else:
        Ynew = overall + schur_prod * (Cyd @ gainR)

    DupdateK = np.exp(Ynew[0:n_active, :])
    sgz1 = np.exp(Ynew[n_active:2 * n_active, :])
    sgsim2 = Ynew[2 * n_active:3 * n_active, :]
    return DupdateK, sgsim2, sgz1
