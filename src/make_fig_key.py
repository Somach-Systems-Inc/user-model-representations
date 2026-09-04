"""Key figure for the 2026-09 write-up (review/fig_key.png).

(a) Linear probe vs. bag-of-words on unseen cue families, four datasets x three
    scales. The probe rises with scale on every dataset; text is flat by
    construction; whether the probe clears text depends on what the NEUTRAL
    class leaks (the holdout only removes the lonely vocabulary).
(b) Steering null: judged behaviour (V+E+D, 0-12) vs. alpha, probe direction
    against a matched-norm random direction, L31, n=10 neutral convos per cell.
(c) Steering null: steered self-report (1-10) vs. alpha, n~85 per cell.

Palette: dataviz slots 1-2 (#2a78d6 blue, #eb6834 orange) + neutral gray; the
pair was validated with scripts/validate_palette.js on a white surface.
Run:  uv run python review/fig_key.py   (from the repo root)
"""
import json, os, sys, collections, statistics as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "review", "fig_key.png")

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
SCALES = ["4B", "9B", "27B"]

# ---- (a) data: unseen-family mean AUC -----------------------------------------
# label lines: dataset, then what changed / what the neutral side leaks (<= 6 words)
DATASETS = [
    dict(key="v1", label="v1\nneutrals name\na friend", probe={"4B": 0.923, "9B": 0.963, "27B": 0.985}, bow=0.948, n=400, bow_pos=("right", "below")),
    dict(key="v1c", label="v1 cleaned\n59 explicit\npairs removed", probe={"4B": 0.867, "9B": 0.929, "27B": 0.975}, bow=0.940, n=282, bow_pos=("left", "above")),
    dict(key="v2", label="v2\nspec-sheet\nregister leaks", probe={"4B": 0.996, "9B": 0.996, "27B": 0.996}, bow=0.995, n=400, bow_pos=("left", "below")),
    dict(key="v3", label="v3\naside list\nnever held out", probe={"4B": 0.943, "9B": 0.945, "27B": 0.981}, bow=0.997, n=400, bow_pos=("left", "above")),
]
# v4 (symmetric holdout) slot: taken from out/fig_datasets.json if its probe dict is filled.
try:
    for d in json.load(open(os.path.join(ROOT, "out", "fig_datasets.json"))):
        if d["label"].startswith("v4") and d.get("probe") and d.get("bow") is not None:
            DATASETS.append(dict(key="v4", label="v4\nboth sides\nheld out", probe=d["probe"], bow=d["bow"], n=d.get("n", 400)))
except FileNotFoundError:
    pass

# ---- (b)/(c) data: steering, recomputed from raw judge / self-report outputs -------
def load_jsonl(p):
    return [json.loads(l) for l in open(p)]

judged = collections.defaultdict(list)
for r in load_jsonl(os.path.join(ROOT, "out", "judged.jsonl")):
    if r["layer"] != 31:
        continue
    s = r["scores"]; judged[(r["direction"], r["alpha"])].append(s["V"] + s["E"] + s["D"])
report = collections.defaultdict(list)
for r in load_jsonl(os.path.join(ROOT, "out", "steered_report_big.jsonl")):
    if r.get("rating") is not None:
        report[r["alpha"]].append(r["rating"])

def mean_se(xs):
    return st.mean(xs), (st.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0)

# ---- figure -------------------------------------------------------------------
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": AXIS,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
fig = plt.figure(figsize=(6.5, 4.0), dpi=200, facecolor="white")
gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.55, 1], height_ratios=[1, 1],
                       wspace=0.36, hspace=0.55, left=0.115, right=0.975, top=0.93, bottom=0.2)
axA = fig.add_subplot(gs[:, 0]); axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[1, 1])
for ax in (axA, axB, axC):
    ax.set_facecolor("white")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6, labelsize=7.5)

# (a) probe vs text, per dataset group ------------------------------------------
GW = 1.0; step = 0.28; ms = {"4B": 4.5, "9B": 5.5, "27B": 6.5}
YMIN = 0.84
for gi, d in enumerate(DATASETS):
    x0 = gi * GW
    xs = [x0 - step, x0, x0 + step]
    ys = [d["probe"][s] for s in SCALES]
    axA.plot([x0 - step - 0.12, x0 + step + 0.12], [d["bow"]] * 2, color=GRAY, lw=2, ls=(0, (3, 2)), zorder=2, solid_capstyle="butt")
    axA.plot(xs, ys, color=BLUE, lw=2, zorder=3, solid_joinstyle="round")
    for x, y, s in zip(xs, ys, SCALES):
        axA.plot(x, y, "o", color=BLUE, ms=ms[s], mec="white", mew=1.2, zorder=4)
    # selective direct labels: 4B and 27B endpoints, plus the text baseline
    lo_above = ys[0] > d["bow"]
    axA.annotate(f"{ys[0]:.3f}", (xs[0], ys[0]), xytext=(0, 5 if lo_above else -5), textcoords="offset points",
                 ha="center", va="bottom" if lo_above else "top", fontsize=6.5, color=INK2)
    hi_above = ys[2] >= d["bow"] - 0.004
    if hi_above:
        axA.annotate(f"{ys[2]:.3f}", (xs[2], ys[2]), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color=INK2)
    else:
        axA.annotate(f"{ys[2]:.3f}", (xs[2], ys[2]), xytext=(6, -1), textcoords="offset points", ha="left", va="center", fontsize=6.5, color=INK2)
    # text baseline value sits just under/over its dashed segment, left end
    side, vert = d.get("bow_pos", ("left", "above" if ys[0] < d["bow"] else "below"))
    bx = x0 - step - 0.12 if side == "left" else x0 + step + 0.12
    axA.annotate(f"text {d['bow']:.3f}", (bx, d["bow"]), xytext=(0, 4 if vert == "above" else -4), textcoords="offset points",
                 ha=side, va="bottom" if vert == "above" else "top", fontsize=6.3, color=MUTED)
    if gi == 0:
        for x, s in zip(xs, SCALES):
            axA.annotate(s, (x, YMIN), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=6.3, color=MUTED)
axA.set_xticks([gi * GW for gi in range(len(DATASETS))], [d["label"] for d in DATASETS], fontsize=6.8, color=INK, linespacing=1.15)
axA.set_xlim(-0.6, (len(DATASETS) - 1) * GW + 0.6)
axA.set_ylim(YMIN, 1.008); axA.set_yticks([0.85, 0.90, 0.95, 1.00])
axA.set_ylabel("mean AUC, held-out cue family\n(axis from 0.84; chance = 0.50)", fontsize=7.5, color=INK2)
axA.set_title("a   Probe vs. bag-of-words on unseen cue families", loc="left", fontsize=8.2, color=INK, fontweight="bold", pad=5)
axA.legend(handles=[Line2D([], [], color=BLUE, lw=2, marker="o", ms=5, mec="white", label="probe, 4B → 9B → 27B"),
                    Line2D([], [], color=GRAY, lw=2, ls=(0, (3, 2)), label="TF-IDF bag-of-words, same text")],
           loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=6.8, frameon=False, handlelength=2.2, borderaxespad=0.2)
axA.tick_params(axis="x", length=0)

# (b) judged behaviour vs alpha ---------------------------------------------------
ALPHAS = [-12, -6, 0, 6, 12]
for key, col, lab, off in (("probe", BLUE, "probe direction", -0.35), ("random", ORANGE, "random direction (matched norm)", 0.35)):
    xs, ys, es = [], [], []
    for a in ALPHAS:
        v = judged.get((key, float(a))) or (judged.get(("probe", 0.0)) if a == 0 else None)  # alpha=0 is unsteered, shared
        if v:
            m, e = mean_se(v); xs.append(a + off); ys.append(m); es.append(e)
    axB.errorbar(xs, ys, yerr=es, color=col, lw=1.6, marker="o", ms=4.5, mec="white", mew=1, capsize=0, elinewidth=0.9, zorder=3)
axB.set_xticks(ALPHAS); axB.set_xlim(-15, 15); axB.set_ylim(0, 4.5); axB.set_yticks([0, 1, 2, 3, 4])
axB.set_ylabel("judged V+E+D\n(0–12 scale)", fontsize=7.2, color=INK2)
axB.set_title("b   Judged behaviour vs. α", loc="left", fontsize=8.2, color=INK, fontweight="bold", pad=4)
axB.legend(handles=[Line2D([], [], color=BLUE, lw=1.6, marker="o", ms=4, mec="white", label="probe direction"),
                    Line2D([], [], color=ORANGE, lw=1.6, marker="o", ms=4, mec="white", label="random direction")],
           loc="lower left", fontsize=6.5, frameon=False, ncol=1, handlelength=1.6, labelspacing=0.2, borderaxespad=0.1)
axB.text(0.98, 0.95, "L31 · n = 10 per cell · ±SE", transform=axB.transAxes, ha="right", va="top", fontsize=6.3, color=MUTED)

# (c) self-report vs alpha ---------------------------------------------------------
xs = sorted(report); ys, es = zip(*(mean_se(report[a]) for a in xs))
axC.errorbar(xs, ys, yerr=es, color=BLUE, lw=1.6, marker="o", ms=4.5, mec="white", mew=1, capsize=0, elinewidth=0.9, zorder=3)
for a, y in zip(xs, ys):
    axC.annotate(f"{y:.2f}", (a, y), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color=INK2)
axC.set_xticks(ALPHAS); axC.set_xlim(-15, 15); axC.set_ylim(1, 7); axC.set_yticks([1, 3, 5, 7])
axC.set_ylabel("steered self-report\n(1–10 scale)", fontsize=7.2, color=INK2)
axC.set_xlabel("steering α along the unit direction\n(±12 ≈ 7.5× the natural class gap)", fontsize=6.8, color=INK2)
axC.set_title("c   Self-report vs. α", loc="left", fontsize=8.2, color=INK, fontweight="bold", pad=4)
n_c = {a: len(report[a]) for a in xs}
axC.text(0.98, 0.95, f"n = {min(n_c.values())}–{max(n_c.values())} per cell · ±SE", transform=axC.transAxes, ha="right", va="top", fontsize=6.3, color=MUTED)

fig.savefig(OUT, dpi=200, facecolor="white")
print("saved", OUT)
