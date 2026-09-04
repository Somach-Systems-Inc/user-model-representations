"""Figure 1 for the 2026-09-03 write-up: probe vs. bag-of-words on unseen cue families,
per dataset generation and model scale. Fill V4 once data/acts_fam_v4_*.pt exist.
Palette: 2 series + neutral baseline (dataviz-validated in make_fig_scale.py)."""
import json, sys
import matplotlib.pyplot as plt

INK = "#1a1f26"; MUTED = "#5c6672"; GRID = "#e4e8ee"
BLUE = "#2a78d6"; GRAY = "#8a93a0"
# {dataset: (label, {scale: probe}, bow)}
DATA = json.load(open("out/fig_datasets.json")) if len(sys.argv) < 2 else json.load(open(sys.argv[1]))
sets = [d for d in DATA if d["probe"]]
scales = ["4B", "9B", "27B"]
fig, axes = plt.subplots(1, len(sets), figsize=(2.6 * len(sets) + 1, 4.2), dpi=200, sharey=True)
if len(sets) == 1: axes = [axes]
fig.patch.set_facecolor("white")
for ax, d in zip(axes, sets):
    ax.set_facecolor("white")
    ax.axhline(d["bow"], color=GRAY, lw=2, ls=(0, (5, 4)), zorder=1)
    xs = [i for i, s in enumerate(scales) if s in d["probe"]]
    ys = [d["probe"][scales[i]] for i in xs]
    ax.plot(xs, ys, color=BLUE, lw=2, marker="o", ms=7, zorder=3)
    for xi, yi in zip(xs, ys):
        ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, color=INK)
    near = any(abs(y - d["bow"]) < 0.02 for y in ys[-1:])  # last probe point close to the line
    ax.text(2.3, d["bow"] + (0.012 if near else 0), f"text {d['bow']:.3f}", color=MUTED, fontsize=8,
            va="bottom" if near else "center", ha="left")
    ax.set_title(d["label"], fontsize=10, color=INK, loc="left")
    ax.set_xticks([0, 1, 2], scales, fontsize=9, color=INK); ax.set_xlim(-0.4, 3.3)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
axes[0].set_ylim(0.80, 1.01); axes[0].set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
axes[0].set_ylabel("unseen-family AUC (axis from 0.80; chance 0.5)", fontsize=9, color=MUTED)
fig.suptitle("Probe (blue) vs. bag-of-words on the same text (gray), by dataset and model scale", fontsize=11, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("out/fig_datasets.png", bbox_inches="tight"); print("saved out/fig_datasets.png")
