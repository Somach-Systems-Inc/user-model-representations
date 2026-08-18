"""The exec-summary opening figure: representation generality scales; the lexical
ceiling can't. Palette validated via dataviz validator (2 series + neutral baseline).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

INK = "#1a1f26"
MUTED = "#5c6672"
GRID = "#e4e8ee"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#8a93a0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures/scale-trend.png")
    args = parser.parse_args()

    models = ["4B", "9B", "27B"]
    x = [0, 1, 2]
    probe = [0.923, 0.963, 0.985]
    transfer_x, transfer = [0, 2], [0.809, 0.984]
    bow = 0.948

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.axhline(bow, color=GRAY, lw=2, ls=(0, (5, 4)), zorder=1)
    ax.text(
        2.02,
        bow - 0.008,
        "bag-of-words ceiling\n(same text, cannot scale)",
        color=MUTED,
        fontsize=9,
        va="top",
        ha="left",
    )

    ax.plot(
        x,
        probe,
        color=BLUE,
        lw=2,
        marker="o",
        ms=8,
        zorder=3,
        label="probe — unseen cue families",
    )
    for xi, yi in zip(x, probe):
        ax.annotate(
            f"{yi:.3f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
            color=INK,
        )

    ax.plot(
        transfer_x,
        transfer,
        color=ORANGE,
        lw=2,
        ls=(0, (2, 3)),
        marker="s",
        ms=8,
        zorder=3,
        label="probe — unseen generator (transfer)",
    )
    for xi, yi in zip(transfer_x, transfer):
        ax.annotate(
            f"{yi:.3f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            fontsize=10,
            color=INK,
        )

    ax.set_xticks(x, [f"Qwen3.5-{model}" for model in models], fontsize=11, color=INK)
    ax.set_ylim(0.78, 1.005)
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.set_ylabel(
        "held-out AUC (axis starts at 0.78; chance = 0.5)", fontsize=10, color=MUTED
    )
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.set_title(
        "User-state probes generalize better with scale; word-counting can't",
        fontsize=12,
        color=INK,
        pad=14,
        loc="left",
    )
    ax.legend(loc="lower right", fontsize=9, frameon=False, labelcolor=INK)
    ax.set_xlim(-0.25, 2.85)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
