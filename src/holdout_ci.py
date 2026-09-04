"""Bootstrap confidence intervals for the cue-family holdout (src/holdout_eval.py).

For each (dataset, scale) this reproduces holdout_eval.py exactly -- leave-one-
cue-family-out, probe = logistic regression on residual activations at a layer
chosen on train-side validation (same seed, same rng consumption order, so the
same layer is picked), BoW = TF-IDF 1-2-gram logistic regression on user turns,
splits by pair_id -- and then puts 95% CIs on the held-out numbers.

WHAT IS BOOTSTRAPPED. The *evaluation*, not the training. The probe and the
BoW model are fitted once on the three training families, exactly as in
holdout_eval.py, and their decision scores on the held-out family's test rows
are computed once. The bootstrap then resamples held-out PAIRS with replacement
(B=1000, seed 0) and recomputes AUC on the resampled score vectors. So the CI
answers "how much would this held-out AUC move if the test pairs were a
different draw of the same size from the same family?", and does not account
for variance from re-fitting the model or re-choosing the layer.

Because all scales of one dataset share the same test pairs, one set of
bootstrap indices is drawn per (dataset, family) and shared across scales and
BoW, so probe-minus-BoW and 27B-minus-4B differences are PAIRED bootstraps.
Resampling whole pairs keeps one lonely and one neutral row per draw, so the
AUC is always defined. The 4-family mean CI is the per-replicate mean across
families (independent resamples per family). CIs are percentile intervals.

Usage (from the repo root):  uv run python src/holdout_ci.py
Writes review/ci_results.md. Does not touch WRITEUP.md or holdout_eval.py.
Only numpy / sklearn for the statistics; torch is used solely to read the .pt
activation blobs, as holdout_eval.py does.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
SCALES = ["4B", "9B", "27B"]
# (dataset name, convos path, acts path template, scales)
DATASETS = [
    ("v1", "data/convos_fam.jsonl", "data/acts_fam_{s}.pt", SCALES),
    ("v1-clean", "data/convos_fam_clean.jsonl", "data/acts_fam_{s}.pt", SCALES),
    ("v3", "data/convos_fam_v3.jsonl", "data/acts_fam_v3_{s}.pt", SCALES),
    ("v4", "data/convos_fam_v4.jsonl", "data/acts_fam_v4_{s}.pt", ["4B", "9B"]),
]
# Headline numbers quoted in WRITEUP.md, used only to label the checks below.
WRITEUP_CLAIMS = {
    "v1": {"probe": {"4B": 0.923, "9B": 0.963, "27B": 0.985}, "bow": 0.948},
    "v1-clean": {"probe": {"4B": 0.867, "9B": 0.929, "27B": 0.975}, "bow": 0.940},
    "v3": {"probe": {"4B": 0.943, "9B": 0.945, "27B": 0.981}, "bow": 0.997},
}


def fit_scores(X_tr, y_tr, X_te):
    clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    return clf.decision_function(X_te)


def load_rows(convos_path, ids_in_blob):
    rows = [json.loads(l) for l in open(convos_path)]
    fam_by_pair = {r["pair_id"]: r["cue_family"] for r in rows if "cue_family" in r}
    rows = [r for r in rows if r["id"] in ids_in_blob]
    return rows, fam_by_pair


def holdout_scores(rows, fam_by_pair, acts, id_to_idx, seed, want_bow):
    """Replicates holdout_eval.main() and returns per-family test-side scores.

    Returns dict F -> {"probe": scores, "bow": scores|None, "y": labels,
    "pairs": pair ids, "layer": int}, with rows in the same order as `rows`
    restricted to the held-out family.
    """
    y_all = np.array([r["label"] != "neutral" for r in rows])
    fam_all = np.array([fam_by_pair[r["pair_id"]] for r in rows])
    idx = np.array([id_to_idx[r["id"]] for r in rows])
    A = acts[idx]
    texts = np.array(["\n".join(m["content"] for m in r["messages"]
                                if m["role"] == "user") for r in rows])
    pair_ids = np.array([r["pair_id"] for r in rows])

    families = sorted(set(fam_all))
    n_layers = A.shape[1]
    rng = random.Random(seed)  # same consumption order as holdout_eval.py
    out = {}
    for F in families:
        te = fam_all == F
        tr = ~te
        tr_pairs = sorted(set(pair_ids[tr]))
        val_pairs = set(rng.sample(tr_pairs, max(1, len(tr_pairs) // 4)))
        val = tr & np.isin(pair_ids, sorted(val_pairs))
        fit = tr & ~val
        layer_scores = [roc_auc_score(y_all[val],
                                      fit_scores(A[fit, L], y_all[fit], A[val, L]))
                        for L in range(n_layers)]
        best_L = int(np.argmax(layer_scores))
        probe = fit_scores(A[tr, best_L], y_all[tr], A[te, best_L])

        bow = None
        if want_bow:
            vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
            X_tr = vec.fit_transform(texts[tr])
            X_te = vec.transform(texts[te])
            bow = fit_scores(X_tr, y_all[tr], X_te)
        out[F] = {"probe": probe, "bow": bow, "y": y_all[te],
                  "pairs": pair_ids[te], "layer": best_L}
    return out


def pair_row_matrix(pairs):
    """[n_pairs, 2] row indices, one row per pair member."""
    d = {}
    for i, p in enumerate(pairs):
        d.setdefault(p, []).append(i)
    mat = np.array([d[p] for p in sorted(d)])
    assert mat.shape[1] == 2, "expected exactly two rows (lonely, neutral) per pair"
    return mat


def boot_auc(y, scores, pair_rows, draws):
    """AUC on each bootstrap draw. draws: [B, n_pairs] indices into pair_rows."""
    out = np.empty(len(draws))
    for b, d in enumerate(draws):
        rows = pair_rows[d].ravel()
        out[b] = roc_auc_score(y[rows], scores[rows])
    return out


def ci(v, lo=2.5, hi=97.5):
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def fmt(point, iv):
    return f"{point:.3f} [{iv[0]:.3f}, {iv[1]:.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0, help="holdout seed (as holdout_eval.py)")
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--out", default="review/ci_results.md")
    args = ap.parse_args()
    B = args.B

    skipped = []
    table = []      # rows for the printed / written table
    per_ds = {}     # dataset -> {"families":[...], "boot": {scale: {F: auc vec}}, "bow": {F: vec}, ...}
    t0 = time.time()

    for name, convos_rel, acts_tpl, scales in DATASETS:
        convos = REPO / convos_rel
        if not convos.exists():
            skipped.append(f"{name}: {convos_rel} missing")
            continue
        avail = [(s, REPO / acts_tpl.format(s=s)) for s in scales]
        missing = [s for s, p in avail if not p.exists()]
        avail = [(s, p) for s, p in avail if p.exists()]
        for s in missing:
            skipped.append(f"{name} @ {s}: {acts_tpl.format(s=s)} missing")
        if not avail:
            continue

        ds = {"scales": [], "point": {}, "boot": {}, "layer": {}, "n_pairs": {}}
        brng = np.random.default_rng(args.boot_seed)
        draws = None  # per family, shared across scales
        for s, acts_path in avail:
            t1 = time.time()
            blob = torch.load(acts_path)
            id_to_idx = {i: k for k, i in enumerate(blob["ids"])}
            acts = blob["acts"].float().numpy()
            rows, fam_by_pair = load_rows(convos, id_to_idx)
            want_bow = "bow" not in ds["point"]
            sc = holdout_scores(rows, fam_by_pair, acts, id_to_idx, args.seed, want_bow)
            del acts, blob
            fams = sorted(sc)
            if draws is None:
                ds["families"] = fams
                draws = {}
                for F in fams:
                    pr = pair_row_matrix(sc[F]["pairs"])
                    draws[F] = (pr, brng.integers(0, len(pr), size=(B, len(pr))))
                    ds["n_pairs"][F] = len(pr)
            if want_bow:
                ds["point"]["bow"] = {F: roc_auc_score(sc[F]["y"], sc[F]["bow"]) for F in fams}
                ds["boot"]["bow"] = {F: boot_auc(sc[F]["y"], sc[F]["bow"], *draws[F]) for F in fams}
            ds["scales"].append(s)
            ds["point"][s] = {F: roc_auc_score(sc[F]["y"], sc[F]["probe"]) for F in fams}
            ds["boot"][s] = {F: boot_auc(sc[F]["y"], sc[F]["probe"], *draws[F]) for F in fams}
            ds["layer"][s] = {F: sc[F]["layer"] for F in fams}
            print(f"[{name} @ {s}] holdout + bootstrap done in {time.time()-t1:.0f}s", flush=True)
        per_ds[name] = ds

        # assemble table rows
        for s in ds["scales"]:
            for F in ds["families"] + ["MEAN"]:
                if F == "MEAN":
                    p = np.mean([ds["point"][s][f] for f in ds["families"]])
                    b = np.mean([ds["point"]["bow"][f] for f in ds["families"]])
                    pv = np.mean([ds["boot"][s][f] for f in ds["families"]], axis=0)
                    bv = np.mean([ds["boot"]["bow"][f] for f in ds["families"]], axis=0)
                    n = sum(ds["n_pairs"].values())
                    layer = ""
                else:
                    p, b = ds["point"][s][F], ds["point"]["bow"][F]
                    pv, bv = ds["boot"][s][F], ds["boot"]["bow"][F]
                    n = ds["n_pairs"][F]
                    layer = ds["layer"][s][F]
                table.append(dict(dataset=name, scale=s, family=F, layer=layer,
                                  probe=p, probe_ci=ci(pv), bow=b, bow_ci=ci(bv),
                                  diff=p - b, diff_ci=ci(pv - bv), n_pairs=n))

    # ---- print ----
    hdr = (f"{'dataset':<9} {'scale':<5} {'family':<12} {'L':>3} {'probe AUC [95% CI]':<24} "
           f"{'BoW AUC [95% CI]':<24} {'probe-BoW [95% CI]':<26} {'n_pairs':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in table:
        print(f"{r['dataset']:<9} {r['scale']:<5} {r['family']:<12} {str(r['layer']):>3} "
              f"{fmt(r['probe'], r['probe_ci']):<24} {fmt(r['bow'], r['bow_ci']):<24} "
              f"{fmt(r['diff'], r['diff_ci']):<26} {r['n_pairs']:>7}")
    for s in skipped:
        print("SKIPPED:", s)
    print(f"\ntotal {time.time()-t0:.0f}s")

    # ---- interpretation (data-driven) ----
    lines = []
    def add(x=""):
        lines.append(x)

    def sig(iv):
        return "excludes 0" if (iv[0] > 0 or iv[1] < 0) else "includes 0"

    add("# Bootstrap 95% CIs for the cue-family holdout")
    add()
    add(f"Generated by `src/holdout_ci.py` (B={B}, bootstrap seed {args.boot_seed}, "
        f"holdout seed {args.seed}). Point estimates reproduce `src/holdout_eval.py` "
        f"exactly (same layer selection, same splits).")
    add()
    add("**What is bootstrapped.** The evaluation, not the training. The probe (logistic "
        "regression on residual activations at the train-side-selected layer) and the BoW "
        "baseline (TF-IDF 1-2-gram logistic regression on user turns) are each fitted once "
        "on the three training families, exactly as in `holdout_eval.py`, and their decision "
        "scores on the held-out family are computed once. Held-out PAIRS are then resampled "
        "with replacement and AUC is recomputed on the resampled score vectors. Fitted models "
        "and the chosen layer stay fixed, so the CI reflects test-set sampling variance only, "
        "not model-refit or layer-choice variance. Resampling whole pairs keeps one lonely and "
        "one neutral row per draw, so AUC is always defined.")
    add()
    add("**Pairing.** For a given dataset every scale is evaluated on the same held-out pairs, "
        "so one set of bootstrap draws is shared across scales and BoW per family. The "
        "probe-minus-BoW and scale-minus-scale intervals below are therefore paired "
        "bootstraps, which is the right test for 'probe > text' and 'bigger model > smaller "
        "model' on the same data; comparing two marginal CIs for overlap would be far too "
        "conservative. The 4-family MEAN is the per-replicate mean of the four family AUCs. "
        "Intervals are percentile intervals.")
    add()
    if skipped:
        add("**Skipped:** " + "; ".join(skipped) + ".")
        add()
    add("## Table")
    add()
    add("| dataset | scale | family | layer | probe AUC [95% CI] | BoW AUC [95% CI] | probe - BoW [95% CI] | n_pairs |")
    add("|---|---|---|---|---|---|---|---|")
    for r in table:
        fam = f"**{r['family']}**" if r["family"] == "MEAN" else r["family"]
        add(f"| {r['dataset']} | {r['scale']} | {fam} | {r['layer']} | {fmt(r['probe'], r['probe_ci'])} | "
            f"{fmt(r['bow'], r['bow_ci'])} | {fmt(r['diff'], r['diff_ci'])} | {r['n_pairs']} |")
    add()

    add("## Paired scale differences (4-family mean probe AUC)")
    add()
    add("| dataset | comparison | difference [95% CI] | verdict |")
    add("|---|---|---|---|")
    scale_verdicts = {}
    for name, ds in per_ds.items():
        fams = ds["families"]
        means = {s: np.mean([ds["boot"][s][f] for f in fams], axis=0) for s in ds["scales"]}
        pts = {s: np.mean([ds["point"][s][f] for f in fams]) for s in ds["scales"]}
        for i in range(len(ds["scales"])):
            for j in range(i + 1, len(ds["scales"])):
                a, b = ds["scales"][i], ds["scales"][j]
                d = means[b] - means[a]
                iv = ci(d)
                verdict = "supported at 95%" if iv[0] > 0 else ("reversed at 95%" if iv[1] < 0 else "within noise")
                scale_verdicts[(name, a, b)] = (pts[b] - pts[a], iv, verdict)
                add(f"| {name} | {b} - {a} | {fmt(pts[b]-pts[a], iv)} | {verdict} |")
    add()

    add("## Interpretation against WRITEUP.md")
    add()
    mean_rows = {(r["dataset"], r["scale"]): r for r in table if r["family"] == "MEAN"}
    fam_rows = {(r["dataset"], r["scale"], r["family"]): r for r in table if r["family"] != "MEAN"}

    def fams_excluding_zero(name, s, sign):
        fams = per_ds[name]["families"]
        hit = [F for F in fams
               if (fam_rows[(name, s, F)]["diff_ci"][0] > 0 if sign > 0
                   else fam_rows[(name, s, F)]["diff_ci"][1] < 0)]
        miss = [F for F in fams if F not in hit]
        return hit, miss

    add("### Claim-by-claim")
    add()
    add("Each WRITEUP.md sentence, then what the paired 95% CI says.")
    add()

    # 1. v1 27B beats text on every family (WRITEUP §3 / RESULTS §6)
    if ("v1", "27B") in mean_rows:
        r = mean_rows[("v1", "27B")]
        hit, miss = fams_excluding_zero("v1", "27B", +1)
        add(f"1. *\"At 27B the probe beats the text baseline on every held-out family\"* (v1). "
            f"Mean: {fmt(r['diff'], r['diff_ci'])} -> {'supported' if r['diff_ci'][0] > 0 else 'within noise'}. "
            f"Per family, the probe-BoW CI excludes 0 for {len(hit)}/4 ({', '.join(hit)}); "
            f"{'within noise for ' + ', '.join(miss) if miss else 'all four clear'}. "
            f"The mean claim holds; the every-family claim is a point-estimate ordering, not a per-family result.")
    # 2. 9B edges out text on v1 (RESULTS §5)
    if ("v1", "9B") in mean_rows:
        r = mean_rows[("v1", "9B")]
        add(f"2. *\"At 9B the probe edges out the text baseline\"* (v1, 0.963 vs 0.948). "
            f"Paired diff {fmt(r['diff'], r['diff_ci'])} -> "
            f"{'supported' if r['diff_ci'][0] > 0 else 'within noise; not a supported win'}.")
    # 3. cleaned: 4B and 9B fall below text; 27B holds
    if ("v1-clean", "27B") in mean_rows:
        r4, r9, r27 = (mean_rows[("v1-clean", s)] for s in ["4B", "9B", "27B"])
        hit, miss = fams_excluding_zero("v1-clean", "27B", +1)
        add(f"3. *\"Dropping the explicit pairs: 4B and 9B fall below text (0.867, 0.929 vs 0.940); 27B holds (0.975)\"*. "
            f"4B below text: {fmt(r4['diff'], r4['diff_ci'])} -> {'supported' if r4['diff_ci'][1] < 0 else 'within noise'}. "
            f"9B below text: {fmt(r9['diff'], r9['diff_ci'])} -> {'supported' if r9['diff_ci'][1] < 0 else 'within noise'}. "
            f"27B above text: {fmt(r27['diff'], r27['diff_ci'])} -> {'supported' if r27['diff_ci'][0] > 0 else 'within noise'} "
            f"(141 pairs). RESULTS.md adds that 27B clears text on all four families here: per family the CI excludes 0 "
            f"only for {', '.join(hit) if hit else 'none'}; {', '.join(miss)} are within noise "
            f"(no_contact has 9 pairs).")
    # 4. v3: probe off the ceiling, text on it
    if ("v3", "27B") in mean_rows:
        parts = []
        for s in per_ds["v3"]["scales"]:
            r = mean_rows[("v3", s)]
            parts.append(f"{s} {fmt(r['diff'], r['diff_ci'])}")
        allneg = all(mean_rows[("v3", s)]["diff_ci"][1] < 0 for s in per_ds["v3"]["scales"])
        add(f"4. *\"v3: the probe comes off the ceiling while text stays on it, but it still loses\"* "
            f"(0.943 / 0.945 / 0.981 vs 0.997). Paired probe-BoW: {'; '.join(parts)} -> "
            f"{'probe significantly below text at every scale; supported' if allneg else 'not below text at every scale'}. "
            f"BoW's own CI ({fmt(mean_rows[('v3','4B')]['bow'], mean_rows[('v3','4B')]['bow_ci'])}) is pinned at the ceiling.")
    # 5. monotone scale trend on every dataset
    parts, all_steps_ok, ends_ok, flat = [], [], [], []
    for name in per_ds:
        steps = [(a, b) for (n, a, b) in scale_verdicts if n == name and
                 per_ds[name]["scales"].index(b) == per_ds[name]["scales"].index(a) + 1]
        ok = [f"{b}>{a}" for a, b in steps if scale_verdicts[(name, a, b)][2] == "supported at 95%"]
        noise = [f"{b}>{a}" for a, b in steps if scale_verdicts[(name, a, b)][2] != "supported at 95%"]
        ends = None
        if len(per_ds[name]["scales"]) >= 2:
            a, b = per_ds[name]["scales"][0], per_ds[name]["scales"][-1]
            ends = scale_verdicts[(name, a, b)]
            if ends[2] == "supported at 95%":
                ends_ok.append(name)
        if steps and not noise:
            all_steps_ok.append(name)
        flat += [f"{name} {x}" for x in noise]
        s = f"{name}: adjacent steps supported {', '.join(ok) if ok else 'none'}"
        if noise:
            s += f"; within noise {', '.join(noise)}"
        if ends:
            s += f"; end-to-end {b}-{a} {fmt(ends[0], ends[1])} ({ends[2]})"
        parts.append(s)
    add(f"5. *\"The probe's unseen-family generalization grows monotonically with scale on every dataset\"*. "
        + " | ".join(parts) + f". End-to-end (smallest to largest) gain supported on "
        f"{', '.join(ends_ok) if ends_ok else 'no dataset'} of {len(per_ds)}; every adjacent step supported on "
        f"{', '.join(all_steps_ok) if all_steps_ok else 'no dataset'}"
        + (f"; flat steps: {', '.join(flat)}." if flat else "."))
    add()

    add("### 'Probe > text' claims (4-family mean, paired probe - BoW)")
    add()
    for name, ds in per_ds.items():
        for s in ds["scales"]:
            r = mean_rows[(name, s)]
            iv = r["diff_ci"]
            if iv[0] > 0:
                v = "probe > text supported at 95%"
            elif iv[1] < 0:
                v = "text > probe at 95% (probe significantly below text)"
            else:
                v = "within noise"
            add(f"- {name} @ {s}: probe {r['probe']:.3f} vs BoW {r['bow']:.3f}, "
                f"diff {fmt(r['diff'], iv)} -> {v}.")
    add()

    add("### The two results the write-up leans on")
    add()
    r = mean_rows.get(("v1-clean", "27B"))
    if r is not None:
        fam_lines = []
        for F in per_ds["v1-clean"]["families"]:
            rr = next(x for x in table if x["dataset"] == "v1-clean" and x["scale"] == "27B" and x["family"] == F)
            fam_lines.append(f"{F} {fmt(rr['diff'], rr['diff_ci'])} ({sig(rr['diff_ci'])}, n={rr['n_pairs']})")
        add(f"- **v1-cleaned, 27B (WRITEUP: 0.975 vs 0.940).** Probe {fmt(r['probe'], r['probe_ci'])}, "
            f"BoW {fmt(r['bow'], r['bow_ci'])}, paired difference {fmt(r['diff'], r['diff_ci'])} "
            f"({sig(r['diff_ci'])}, {r['n_pairs']} pairs). Per family: " + "; ".join(fam_lines) + ".")
    else:
        add("- v1-cleaned @ 27B: not available (skipped).")
    for s in ["4B", "9B", "27B"]:
        r = mean_rows.get(("v3", s))
        if r is not None:
            add(f"- **v3, {s} (WRITEUP: probe {WRITEUP_CLAIMS['v3']['probe'][s]:.3f} vs text 0.997).** "
                f"Probe {fmt(r['probe'], r['probe_ci'])}, BoW {fmt(r['bow'], r['bow_ci'])}, paired difference "
                f"{fmt(r['diff'], r['diff_ci'])} ({sig(r['diff_ci'])}).")
    add()

    add("### 'Scale trend' claims (paired, same test pairs across scales)")
    add()
    for name in per_ds:
        parts = []
        for (n, a, b), (d, iv, v) in scale_verdicts.items():
            if n == name:
                parts.append(f"{b}-{a} {fmt(d, iv)} ({v})")
        if parts:
            add(f"- {name}: " + "; ".join(parts) + ".")
    add()
    add("Reading: a scale step is 'supported' when the paired CI of the mean-AUC difference "
        "lies above 0. A monotone point trend whose individual steps are within noise is a "
        "trend hypothesis, not a demonstrated effect, on this many pairs.")
    add()
    add("### Caveats")
    add()
    add("- These intervals do not include training-side variance (re-fitting on other pairs, "
        "re-choosing the layer). They are a lower bound on total uncertainty.")
    add("- AUCs near 1.0 have asymmetric, bounded intervals; percentile CIs on a mean of four "
        "such families can be narrow on the high side.")
    add("- v1-clean `no_contact` keeps only 9 pairs; its interval is essentially uninformative.")
    add("- The BoW baseline is fixed across scales by construction (same text), so its "
        "interval is identical in every scale row of a dataset.")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
