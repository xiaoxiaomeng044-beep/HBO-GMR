import numpy as np
import pandas as pd


def gethistorydata(timestep, measurements, history, n_producers=6, n_timesteps=82):
    with open("Real.RSM", "rt") as f:
        true1 = pd.read_fwf(f, colspecs="infer", skiprows=6, header=None)

    t_foe = pd.to_numeric(true1.iloc[0:history, 2], errors="coerce")
    t_bhp = [pd.to_numeric(true1.iloc[0:history, 3 + k], errors="coerce") for k in range(n_producers)]
    tg1 = pd.to_numeric(true1.iloc[0:history, 3 + n_producers], errors="coerce")

    with open("Real.RSM", "rt") as f:
        true2 = pd.read_fwf(f, colspecs="infer", skiprows=94, header=None)

    n_gor_remaining = n_producers - 1
    tg_rest = [pd.to_numeric(true2.iloc[0:history, 1 + k], errors="coerce") for k in range(n_gor_remaining)]

    wct_col_start = 1 + n_gor_remaining
    n_wct_block2 = min(true2.shape[1] - wct_col_start, n_producers)
    tw_block2 = [
        pd.to_numeric(true2.iloc[0:history, wct_col_start + k], errors="coerce")
        for k in range(n_wct_block2)
    ]

    n_wct_remaining = n_producers - n_wct_block2
    tw_block3 = []
    if n_wct_remaining > 0:
        with open("Real.RSM", "rt") as f:
            true3 = pd.read_fwf(f, colspecs="infer", skiprows=182, header=None)
        tw_block3 = [
            pd.to_numeric(true3.iloc[0:history, 1 + k], errors="coerce")
            for k in range(n_wct_remaining)
        ]

    observation = np.zeros((measurements, history))
    for ihistory in range(history):
        obs = np.zeros(measurements)
        obs[0] = t_foe.iloc[ihistory]
        for k in range(n_producers):
            obs[1 + k] = t_bhp[k].iloc[ihistory]
        obs[1 + n_producers] = tg1.iloc[ihistory]
        for k in range(n_gor_remaining):
            obs[1 + n_producers + 1 + k] = tg_rest[k].iloc[ihistory]
        for k in range(n_wct_block2):
            obs[1 + 2 * n_producers + k] = tw_block2[k].iloc[ihistory]
        for k in range(n_wct_remaining):
            obs[1 + 2 * n_producers + n_wct_block2 + k] = tw_block3[k].iloc[ihistory]
        observation[:, ihistory] = obs

    return observation
