"""Figures for the ladder, one per file, 600 dpi PNG plus vector PDF (RESEARCH.md checklist).

Reads results/analysis/cells.csv (written by analyze.py). Every interval is the same hierarchical
bootstrap as the tests (subjects, then sessions within subjects), recomputed here from the cells so
the figure and the table cannot disagree. A figure whose inputs do not exist yet is skipped and named.

Form and colour (dataviz skill, palette validated with its script, 2026-10-01: blue #2a78d6 and
orange #eb6834, worst colour-blind dE 24.7, all checks pass):
  * Fig. 1 compares many methods on one quantity: a dot-and-interval row per method, common x axis.
    Two families only -- this project's ladder and published leaderboard entries -- each with its own
    hue AND marker shape, so identity never rests on colour; every row is labelled directly.
  * Figs. 2-4 are single-series: no legend, the title says what is plotted.
Text stays in neutral ink; grid is recessive; marks are thin.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# isort: off
import resources  # noqa: F401,E402  MUST load before numpy: caps BLAS threads
# isort: on

import argparse  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analyze import hier_boot, overall  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OURS, THEIRS = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"

LABEL = {
    "L0": "Time in recording only (L0)", "L1": "Amplitude only (L1)", "L2": "Dynamics only (L2)",
    "L3": "Dynamics + amplitude (L3)", "L4": "Spectral shape only (L4)", "L5": "Spectrogram (L5)",
    "pub_linear_lap_stft": "Linear, published", "brainbert_perwindow": "BrainBERT, per-window",
    "brainbert_untrained_perwindow": "BrainBERT untrained, per-window",
    "brainbert_globalz": "BrainBERT, global z", "popt_globalz": "PopT, global z",
    "popt_perwindow": "PopT, per-window (unpaired)", "diver1_tiny_frozen": "DIVER-1 tiny, frozen",
    "diver1_tiny": "DIVER-1 tiny, fine-tuned", "mapa": "MAPA",
}
TASK = {
    "onset": "Sentence onset", "speech": "Speech", "volume": "Volume", "delta_volume": "Delta volume",
    "pitch": "Voice pitch", "word_index": "Word position", "word_gap": "Inter-word gap",
    "gpt2_surprisal": "GPT-2 surprisal", "word_head_pos": "Head word position",
    "word_part_speech": "Part of speech", "word_length": "Word length",
    "global_flow": "Global optical flow", "local_flow": "Local optical flow",
    "frame_brightness": "Frame brightness", "face_num": "Number of faces",
}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelcolor=INK, width=0.6, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, out, name, data):
    """Write the figure and the numbers it plots (RESEARCH.md section 11: a figure's numbers must exist
    somewhere other than inside the image, so captions and text can be pinned to them)."""
    data.to_json(os.path.join(out, f"{name}.json"), orient="records", indent=1)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {name}", flush=True)


def per_task_diff(cells, a, b, split="within"):
    ca = cells[(cells.model == a) & (cells.split == split)].set_index(["test_subject", "test_trial", "task"]).auroc
    cb = cells[(cells.model == b) & (cells.split == split)].set_index(["test_subject", "test_trial", "task"]).auroc
    d = (ca - cb).dropna()
    rows = []
    for task, g in d.groupby(level="task"):
        m, lo, hi = hier_boot(g.droplevel("task"), f"fig_{a}_{b}_{task}")
        rows.append(dict(task=task, mean=m, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def fig_ladder(cells, out):
    rows = []
    for model in LABEL:
        o = overall(cells, model, "within")
        if len(o) < 2:
            continue
        m, lo, hi = hier_boot(o, f"fig1_{model}")
        rows.append(dict(model=model, mean=m, lo=lo, hi=hi, n=len(o), ours=model.startswith("L")))
    if not rows:
        print("skip fig1: no overall scores", flush=True)
        return
    df = pd.DataFrame(rows).sort_values("mean")
    fig, ax = plt.subplots(figsize=(6.6, 0.28 * len(df) + 0.9))
    y = np.arange(len(df))
    for yi, r in zip(y, df.itertuples()):
        c, mk = (OURS, "o") if r.ours else (THEIRS, "D")
        ax.plot([r.lo, r.hi], [yi, yi], color=c, linewidth=1.4, solid_capstyle="round")
        ax.plot(r.mean, yi, marker=mk, color=c, markersize=5.5, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[m] for m in df.model])
    ax.axvline(0.5, color=INK2, linewidth=0.6, linestyle=(0, (2, 2)))
    ax.set_xlabel("Within-session AUROC, mean over 15 tasks (95% hierarchical bootstrap)", color=INK, fontsize=8)
    ax.plot([], [], "o", color=OURS, label="This study, one pipeline")
    ax.plot([], [], "D", color=THEIRS, label="Neuroprobe leaderboard")
    # Legend outside the plotting area (RESEARCH.md section 11), above the axes.
    ax.legend(frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, labelcolor=INK)
    style(ax)
    save(fig, out, "f01_ladder_overall", df)


def fig_task_diff(cells, a, b, tau, out, name, xlabel):
    d = per_task_diff(cells, a, b)
    if d.empty:
        print(f"skip {name}: needs {a} and {b}", flush=True)
        return
    d["tau"] = d.task.map(tau)
    d = d.sort_values(["tau", "mean"])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    y = np.arange(len(d))
    ax.hlines(y, d.lo, d.hi, color=OURS, linewidth=1.4)
    ax.plot(d["mean"], y, "o", color=OURS, markersize=5, markeredgecolor="white", markeredgewidth=0.8)
    ax.axvline(0, color=INK2, linewidth=0.6, linestyle=(0, (2, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels([f"{TASK[t]}  ({tau.get(t, float('nan')):g} s)" for t in d.task])
    ax.set_xlabel(xlabel, color=INK, fontsize=8)
    style(ax)
    save(fig, out, name, d)


def fig_time_only(cells, tau, out):
    c0 = cells[(cells.model == "L0") & (cells.split == "within")]
    if c0.empty:
        print("skip f04: needs L0", flush=True)
        return
    rows = []
    for task, g in c0.groupby("task"):
        m, lo, hi = hier_boot(g.set_index(["test_subject", "test_trial"]).auroc, f"fig4_{task}")
        rows.append(dict(task=task, mean=m, lo=lo, hi=hi, tau=tau.get(task, np.nan)))
    d = pd.DataFrame(rows).sort_values(["tau", "mean"])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    y = np.arange(len(d))
    ax.hlines(y, d.lo, d.hi, color=OURS, linewidth=1.4)
    ax.plot(d["mean"], y, "o", color=OURS, markersize=5, markeredgecolor="white", markeredgewidth=0.8)
    ax.axvline(0.5, color=INK2, linewidth=0.6, linestyle=(0, (2, 2)))
    ax.set_yticks(y)
    ax.set_yticklabels([f"{TASK[t]}  ({t_:g} s)" for t, t_ in zip(d.task, d.tau)])
    ax.set_xlabel("AUROC from the window's start time alone (no neural data)", color=INK, fontsize=8)
    style(ax)
    save(fig, out, "f04_time_only", d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=os.path.join(HERE, "results", "analysis", "cells.csv"))
    ap.add_argument("--tau", default=os.path.join(HERE, "results", "label_timescale", "tau_by_task.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    cells = pd.read_csv(a.cells)
    tau = pd.read_csv(a.tau).set_index("task").tau_s_median.to_dict() if os.path.exists(a.tau) else {}
    fig_ladder(cells, a.out)
    fig_task_diff(cells, "L5", "L4", tau, a.out, "f02_amplitude_contribution_by_task",
                  "L5 minus L4: AUROC lost when amplitude is removed")
    fig_task_diff(cells, "L2", "L4", tau, a.out, "f03_dynamics_vs_shape_by_task",
                  "L2 minus L4: dynamics battery relative to spectral shape")
    fig_time_only(cells, tau, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
