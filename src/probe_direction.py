"""Fit a family-general probe on cached activations and save a steering direction.

    uv run python src/probe_direction.py --convos data/convos_fam_v4.jsonl \
        --acts data/acts_fam_v4_27B.pt --out out/dir_v4_27B.pt

Layer is chosen by 5-fold pair-grouped CV over all layers; the saved blob matches the
format steer.py expects (direction, layer) and adds the natural class gap so alphas
can be reported in units of the model's own separation.
"""
import argparse, json
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--convos", default="data/convos_fam_v4.jsonl")
ap.add_argument("--acts", default="data/acts_fam_v4_27B.pt")
ap.add_argument("--out", default="out/dir_v4_27B.pt")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rows = {json.loads(l)["id"]: json.loads(l) for l in open(a.convos)}
blob = torch.load(a.acts)
keep = [i for i, cid in enumerate(blob["ids"]) if cid in rows]
X_all = blob["acts"][keep].float().numpy()
ids = [blob["ids"][i] for i in keep]
y = np.array([rows[i]["label"] != "neutral" for i in ids])
groups = np.array([rows[i]["pair_id"] for i in ids])
print(f"{X_all.shape[0]} convos, {X_all.shape[1]} layers, d={X_all.shape[2]}")

best = (None, -1)
for layer in range(1, X_all.shape[1]):
    X = X_all[:, layer]
    aucs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = LogisticRegression(max_iter=2000, random_state=a.seed).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(X[te])))
    m = float(np.mean(aucs))
    if m > best[1]:
        best = (layer, m)
layer, cv_auc = best
print(f"best layer {layer}, CV AUC {cv_auc:.3f}")

X = X_all[:, layer]
clf = LogisticRegression(max_iter=2000, random_state=a.seed).fit(X, y)
d = torch.tensor(clf.coef_[0], dtype=torch.float32)
d = d / d.norm()
proj = X @ d.numpy()
gap = float(proj[y].mean() - proj[~y].mean())
norm = float(np.linalg.norm(X, axis=1).mean())
print(f"natural class gap {gap:.3f}; mean residual norm {norm:.1f}")
torch.save({"direction": d, "layer": layer, "cv_auc": cv_auc,
            "natural_gap": gap, "mean_norm": norm, "kind": "logreg",
            "acts": a.acts, "convos": a.convos}, a.out)
print("saved", a.out)
