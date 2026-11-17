import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from metrics_production import PRODUCTION_VARIABLES, _read_rsm_table


def _safe_std(x, eps=1e-12):
    s = np.nanstd(x)
    return s if np.isfinite(s) and s > eps else 1.0


def _mse_by_ensemble(sim_mat, true_vec):
    return np.nanmean((sim_mat - np.asarray(true_vec, dtype=float).reshape(-1, 1)) ** 2, axis=0)


def _save_single_plot(time, ens_data, true_data, title, ylabel, ylim, save_path):
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for i in range(ens_data.shape[1]):
        ax.plot(
            time,
            ens_data[:, i],
            color="#173f90",
            lw=0.8,
            alpha=0.35,
            label="Realisations" if i == 0 else None,
        )

    ax.plot(time, true_data, color="red", lw=2.0, label="True model", zorder=5)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def getcostfunctionvalue(timestep, N, inelly, save_single_plots=False):
    truth = _read_rsm_table("Real.RSM")
    time = truth["Time"]
    n_time = len(time)
    cwd = os.getcwd()

    ens = {name: np.zeros((n_time, N)) for name in PRODUCTION_VARIABLES}
    for i in range(N):
        master_rsm = os.path.join(cwd, f"MASTER{i + 1:05d}", "MASTER0.RSM")
        sim = _read_rsm_table(master_rsm)
        for name in PRODUCTION_VARIABLES:
            ens[name][:, i] = sim[name]

    if save_single_plots:
        _save_single_plot(
            time, ens["FOE"], truth["FOE"],
            "Field oil recovery ratio", "Oil recovery ratio", (0, 0.25),
            f"FOE_iter{inelly}.jpg",
        )
        _save_single_plot(
            time, ens["WBHP2"], truth["WBHP2"],
            "PRO-4-WBHP", "WBHP (BARSA)", (100, 300),
            f"PRO-4-WBHP_iter{inelly}.jpg",
        )
        _save_single_plot(
            time, ens["GOR4"], truth["GOR4"],
            "PRO-11-GOR", r"Gas oil ratio (Sm$^3$/Sm$^3$)", (0, 350),
            f"PRO-11-GOR_iter{inelly}.jpg",
        )

    total_error = np.zeros(N, dtype=float)
    for prefix in ("WBHP", "WCT", "GOR"):
        for k in range(1, 7):
            key = f"{prefix}{k}"
            total_error += _mse_by_ensemble(ens[key], truth[key]) / _safe_std(truth[key])

    index = total_error / len(timestep)
    best_idx = int(np.argmin(index)) + 1
    print(f"Best production match realization: #{best_idx} (value={np.min(index):.6f})")

    snapshot = {
        "iteration": int(inelly),
        "Time": time.copy(),
        "FOE_ens": ens["FOE"].copy(),
        "FOE_true": truth["FOE"].copy(),
        "WBHP4_ens": ens["WBHP2"].copy(),
        "WBHP4_true": truth["WBHP2"].copy(),
        "GOR11_ens": ens["GOR4"].copy(),
        "GOR11_true": truth["GOR4"].copy(),
    }
    return index, snapshot
