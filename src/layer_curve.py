"""Per-layer probe AUC across model scales: does the signal peak later at 27B?

    uv run python src/layer_curve.py --convos data/convos_fam_v4.jsonl --tag v4

For each scale, 5-fold pair-grouped CV at every residual point. Layer index is also
reported as a fraction of depth so 33-point and 65-point models are comparable.
"""
import argparse, json
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--convos", default="data/convos_fam_v4.jsonl")
ap.add_argument("--acts-tpl", default="data/acts_fam_v4_{s}.pt")
ap.add_argument("--scales", default="4B,9B,27B")
ap.add_argument("--tag", default="v4")
a = ap.parse_args()

rows = {json.loads(l)["id"]: json.loads(l) for l in open(a.convos)}
out = {}
for scale in a.scales.split(","):
    blob = torch.load(a.acts_tpl.format(s=scale))
    keep = [i for i, cid in enumerate(blob["ids"]) if cid in rows]
    X_all = blob["acts"][keep].float().numpy()
    ids = [blob["ids"][i] for i in keep]
    y = np.array([rows[i]["label"] != "neutral" for i in ids])
    g = np.array([rows[i]["pair_id"] for i in ids])
    curve = []
    for layer in range(X_all.shape[1]):
        X = X_all[:, layer]
        aucs = [roc_auc_score(y[te], LogisticRegression(max_iter=2000, random_state=0)
                              .fit(X[tr], y[tr]).decision_function(X[te]))
                for tr, te in GroupKFold(n_splits=5).split(X, y, g)]
        curve.append(float(np.mean(aucs)))
    best = int(np.argmax(curve))
    out[scale] = {"curve": curve, "n_layers": len(curve), "best_layer": best,
                  "best_auc": curve[best], "best_frac": best / (len(curve) - 1)}
    print(f"{scale}: peak {curve[best]:.3f} at layer {best}/{len(curve)-1} "
          f"({out[scale]['best_frac']:.0%} depth)")
json.dump(out, open(f"out/layer_curve_{a.tag}.json", "w"), indent=1)
print(f"saved out/layer_curve_{a.tag}.json")
