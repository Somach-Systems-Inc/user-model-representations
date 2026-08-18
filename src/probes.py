"""Train logistic probes per layer with the controls Neel requires.

Splits are BY PAIR (a lonely/neutral pair never straddles train/test — otherwise the
probe can key on topic phrasing shared within the pair and the AUC is a lie).

Outputs: out/probe_auc_by_layer.png, out/probe_dir.pt (best layer's direction),
and a printed table of probe vs controls.
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def pair_split(ids: list[str], test_frac: float, seed: int) -> np.ndarray:
    pairs = sorted({i.rsplit("_", 1)[0] for i in ids})
    rng = random.Random(seed)
    test_pairs = set(rng.sample(pairs, int(len(pairs) * test_frac)))
    return np.array([i.rsplit("_", 1)[0] in test_pairs for i in ids])


def probe_auc(X_tr, y_tr, X_te, y_te) -> tuple[float, np.ndarray]:
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X_tr, y_tr)
    return roc_auc_score(y_te, clf.decision_function(X_te)), clf.coef_[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="data/acts_Qwen3.5-4B.pt")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args()

    blob = torch.load(args.acts)
    acts = blob["acts"].float().numpy()  # [n, layers+1, d]
    y = np.array([l != "neutral" for l in blob["labels"]])  # any state vs neutral
    state = next((l for l in blob["labels"] if l != "neutral"), "state")
    is_test = pair_split(blob["ids"], args.test_frac, args.seed)
    rng = np.random.default_rng(args.seed)

    n_layers = acts.shape[1]
    aucs, shuf_aucs, rand_aucs, dirs = [], [], [], []
    for layer in range(n_layers):
        X = acts[:, layer]
        auc, direction = probe_auc(X[~is_test], y[~is_test], X[is_test], y[is_test])
        # control 1: shuffled labels — what overfitting alone buys
        y_shuf = rng.permutation(y[~is_test])
        shuf, _ = probe_auc(X[~is_test], y_shuf, X[is_test], y[is_test])
        # control 2: random direction at matched norm, no training
        rd = rng.standard_normal(X.shape[1])
        rand = roc_auc_score(y[is_test], X[is_test] @ rd)
        aucs.append(auc)
        shuf_aucs.append(shuf)
        rand_aucs.append(max(rand, 1 - rand))
        dirs.append(direction)

    best = int(np.argmax(aucs))
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    torch.save(
        {
            "direction": torch.tensor(dirs[best]),
            "layer": best,
            "auc": aucs[best],
            "acts_file": args.acts,
        },
        out / "probe_dir.pt",
    )

    plt.figure(figsize=(7, 4))
    plt.plot(aucs, label="probe")
    plt.plot(shuf_aucs, label="shuffled labels", ls="--")
    plt.plot(rand_aucs, label="random direction", ls=":")
    plt.axhline(0.5, color="gray", lw=0.5)
    plt.xlabel("layer")
    plt.ylabel("test AUC (pair-held-out)")
    plt.title(f"user-{state} probe — {blob['model']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "probe_auc_by_layer.png", dpi=200)

    print(
        json.dumps(
            {
                "best_layer": best,
                "best_auc": round(aucs[best], 4),
                "shuffled_at_best": round(shuf_aucs[best], 4),
                "random_dir_at_best": round(rand_aucs[best], 4),
            },
            indent=2,
        )
    )
    print("REMINDER: run the bag-of-words text baseline before quoting any of this.")


if __name__ == "__main__":
    main()
