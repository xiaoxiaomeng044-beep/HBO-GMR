import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def _style_rcparams():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _default_iter_labels(selected_iters):
    n = len(selected_iters)
    labels = []
    for i, it in enumerate(selected_iters):
        if i == 0:
            labels.append("Prior")
        elif i == n - 1:
            labels.append("Final posterior")
        else:
            labels.append(f"Iter {it}")
    return labels


def _subsample_indices(n_members, max_lines=20):
    if n_members <= max_lines:
        return np.arange(n_members)
    return np.linspace(0, n_members - 1, max_lines, dtype=int)


def _draw_panel(
    ax,
    time,
    ens,
    truth,
    ylim=None,
    mode="hybrid",
    max_lines=20,
    real_color="#90A4C6",
    band_color="#4C78A8",
    p50_color="#1F4E9E",
    truth_color="#D62728",
):
    ens = np.asarray(ens, dtype=float)
    truth = np.asarray(truth, dtype=float)
    time = np.asarray(time, dtype=float)

    p10 = np.nanpercentile(ens, 10, axis=1)
    p50 = np.nanpercentile(ens, 50, axis=1)
    p90 = np.nanpercentile(ens, 90, axis=1)

    if mode in ("lines", "hybrid"):
        for i in _subsample_indices(ens.shape[1], max_lines=max_lines):
            ax.plot(
                time, ens[:, i],
                color=real_color, lw=0.6,
                alpha=0.18 if mode == "hybrid" else 0.28,
                zorder=1,
            )

    if mode in ("band", "hybrid"):
        ax.fill_between(
            time, p10, p90,
            color=band_color,
            alpha=0.12 if mode == "hybrid" else 0.18,
            linewidth=0,
            zorder=2,
        )
        ax.plot(time, p10, color=band_color, lw=0.9, ls="--", alpha=0.9, zorder=3)
        ax.plot(time, p90, color=band_color, lw=0.9, ls="--", alpha=0.9, zorder=3)
        ax.plot(time, p50, color=p50_color, lw=1.8, alpha=1.0, zorder=4)

    ax.plot(time, truth, color=truth_color, lw=2.0, zorder=5)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.grid(True, alpha=0.10, linewidth=0.45)
    ax.set_facecolor("white")


def _build_legend_handles(mode="hybrid"):
    handles = []
    if mode in ("lines", "hybrid"):
        handles.append(Line2D([0], [0], color="#90A4C6", lw=1.2, alpha=0.7, label="Realisations"))
    if mode in ("band", "hybrid"):
        handles.extend([
            Patch(facecolor="#4C78A8", alpha=0.12, edgecolor="none", label="P10-P90"),
            Line2D([0], [0], color="#4C78A8", lw=1.0, ls="--", label="P10 / P90"),
            Line2D([0], [0], color="#1F4E9E", lw=1.8, label="P50"),
        ])
    handles.append(Line2D([0], [0], color="#D62728", lw=2.0, label="True model"))
    return handles


def plot_history_match_figure_advanced(
    history_match_snapshots,
    selected_iters=None,
    iter_labels=None,
    out_png="Figure3_history_match_advanced.png",
    out_pdf="Figure3_history_match_advanced.pdf",
    mode="hybrid",
    max_lines=20,
    figsize=None,
):
    if not history_match_snapshots:
        return

    _style_rcparams()
    all_iters = sorted(history_match_snapshots.keys())
    selected_iters = all_iters if selected_iters is None else [it for it in selected_iters if it in history_match_snapshots]
    if not selected_iters:
        return

    ncol = len(selected_iters)
    if iter_labels is None:
        iter_labels = _default_iter_labels(selected_iters)
    if len(iter_labels) != ncol:
        raise ValueError("len(iter_labels) must match len(selected_iters)")

    row_specs = [
        {"ens_key": "FOE_ens", "true_key": "FOE_true", "ylim": (0.0, 0.25), "ylabel": "Field oil recovery ratio"},
        {"ens_key": "WBHP4_ens", "true_key": "WBHP4_true", "ylim": (100, 300), "ylabel": "PRO-4 WBHP (BARSA)"},
        {"ens_key": "GOR11_ens", "true_key": "GOR11_true", "ylim": (0, 350), "ylabel": r"PRO-11 GOR (Sm$^3$/Sm$^3$)"},
    ]

    if figsize is None:
        figsize = (2.75 * ncol, 6.8)

    fig, axes = plt.subplots(3, ncol, figsize=figsize, sharex="col", squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.09, top=0.88, wspace=0.20, hspace=0.10)

    for c, it in enumerate(selected_iters):
        snap = history_match_snapshots[it]
        time = np.asarray(snap["Time"], dtype=float)
        axes[0, c].set_title(iter_labels[c], pad=10)
        for r, spec in enumerate(row_specs):
            ax = axes[r, c]
            _draw_panel(
                ax=ax,
                time=time,
                ens=snap[spec["ens_key"]],
                truth=snap[spec["true_key"]],
                ylim=spec["ylim"],
                mode=mode,
                max_lines=max_lines,
            )
            ax.set_ylabel(spec["ylabel"] if c == 0 else "")
            if r == 2:
                ax.set_xlabel("Time (days)")
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)

    handles = _build_legend_handles(mode=mode)
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
        columnspacing=1.1,
        handlelength=2.2,
    )
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
