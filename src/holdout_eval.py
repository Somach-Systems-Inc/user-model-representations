"""Cue-family holdout: the experiment that separates 'reads the state' from
'detects the words'. For each family F: train probe and bag-of-words on the other
three families (+ their paired neutrals), test on F. Layer for the probe is chosen
on train-side validation only (no test peeking).

If probe >> BoW on held-out families, the activation direction generalizes across
surface realizations of loneliness and the lexical shortcut does not.
"""

import argparse
import json
import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def probe_auc(X_tr, y_tr, X_te, y_te) -> float:
    clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    return roc_auc_score(y_te, clf.decision_function(X_te))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convos", default="data/convos_fam.jsonl")
    ap.add_argument("--acts", default="data/acts_fam_4B.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.convos, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    fam_by_pair = {r["pair_id"]: r["cue_family"] for r in rows if "cue_family" in r}
    blob = torch.load(args.acts)
    id_to_idx = {i: k for k, i in enumerate(blob["ids"])}
    rows = [r for r in rows if r["id"] in id_to_idx]

    acts = blob["acts"].float().numpy()
    y_all = np.array([r["label"] != "neutral" for r in rows])
    fam_all = np.array([fam_by_pair[r["pair_id"]] for r in rows])
    idx = np.array([id_to_idx[r["id"]] for r in rows])
    acts = acts[idx]
    texts = np.array(
        [
            "\n".join(m["content"] for m in r["messages"] if m["role"] == "user")
            for r in rows
        ]
    )
    pair_ids = np.array([r["pair_id"] for r in rows])

    families = sorted(set(fam_all))
    n_layers = acts.shape[1]
    rng = random.Random(args.seed)
    results = defaultdict(dict)

    for F in families:
        te = fam_all == F
        tr = ~te
        # layer selection on train side only: hold out 25% of train PAIRS
        tr_pairs = sorted(set(pair_ids[tr]))
        val_pairs = set(rng.sample(tr_pairs, max(1, len(tr_pairs) // 4)))
        val = tr & np.isin(pair_ids, sorted(val_pairs))
        fit = tr & ~val
        layer_scores = [
            probe_auc(acts[fit, L], y_all[fit], acts[val, L], y_all[val])
            for L in range(n_layers)
        ]
        best_L = int(np.argmax(layer_scores))
        results[F]["probe"] = probe_auc(
            acts[tr, best_L], y_all[tr], acts[te, best_L], y_all[te]
        )
        results[F]["layer"] = best_L

        vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
        X_tr = vec.fit_transform(texts[tr])
        X_te = vec.transform(texts[te])
        results[F]["bow"] = probe_auc(X_tr, y_all[tr], X_te, y_all[te])

    print(f"{'held-out family':>16} {'probe AUC':>10} {'(layer)':>8} {'BoW AUC':>8}")
    for F in families:
        r = results[F]
        print(f"{F:>16} {r['probe']:>10.3f} {r['layer']:>8} {r['bow']:>8.3f}")
    pm = np.mean([results[F]["probe"] for F in families])
    bm = np.mean([results[F]["bow"] for F in families])
    print(f"{'MEAN':>16} {pm:>10.3f} {'':>8} {bm:>8.3f}")
    print(
        "\nInterpretation: probe >> BoW here = the direction reads the state, "
        "not the vocabulary. Probe ~= BoW = still a word detector; be honest."
    )


if __name__ == "__main__":
    main()
