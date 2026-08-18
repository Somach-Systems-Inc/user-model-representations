"""Generator-transfer evaluation: apply the probe trained on the main (Opus-generated)
dataset to activations from the transfer set (gpt-oss-generated). No retraining —
just direction · activation at the probe's layer. If AUC holds, the probe reads a
representation, not the fingerprints of one generator's writing style.
"""

import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="out/probe_dir.pt")
    ap.add_argument("--acts", required=True, help="acts .pt for the transfer convos")
    args = ap.parse_args()

    blob = torch.load(args.direction)
    d = blob["direction"].numpy()
    layer = blob["layer"]

    acts_blob = torch.load(args.acts)
    X = acts_blob["acts"].float().numpy()[:, layer]
    y = np.array([l != "neutral" for l in acts_blob["labels"]])

    scores = X @ d
    auc = roc_auc_score(y, scores)
    auc = max(auc, 1 - auc)  # direction sign is arbitrary across datasets
    print(
        f"transfer AUC at layer {layer}: {auc:.3f} "
        f"(train-side AUC was {blob['auc']:.3f}; n={len(y)}, "
        f"{int(y.sum())} state / {int((~y).sum())} neutral)"
    )


if __name__ == "__main__":
    main()
