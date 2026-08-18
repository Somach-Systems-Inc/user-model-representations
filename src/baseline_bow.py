"""The embarrassing baseline: bag-of-words logistic regression on raw user-turn text.

Any probe result must beat this to claim the representation is interestingly
"internal" — if unigrams on the transcript match the activation probe, the probe
has only rediscovered the words. Same pair-held-out split as probes.py.
"""

import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from probes import pair_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convos", default="data/convos.jsonl")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--eval-convos",
        help="train on ALL of --convos, test on this file instead "
        "(generator-transfer comparison for the probe)",
    )
    args = ap.parse_args()

    def load(path):
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        texts = [
            "\n".join(m["content"] for m in r["messages"] if m["role"] == "user")
            for r in rows
        ]
        y = np.array([r["label"] != "neutral" for r in rows])
        return rows, texts, y

    rows, texts, y = load(args.convos)
    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
    if args.eval_convos:
        _, texts_te, y_te = load(args.eval_convos)
        X_tr, y_tr = vec.fit_transform(texts), y
    else:
        is_test = pair_split([r["id"] for r in rows], args.test_frac, args.seed)
        X_tr = vec.fit_transform([t for t, m in zip(texts, is_test) if not m])
        y_tr = y[~is_test]
        texts_te, y_te = [t for t, m in zip(texts, is_test) if m], y[is_test]
    X_te = vec.transform(texts_te)
    clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, clf.decision_function(X_te))

    names = np.array(vec.get_feature_names_out())
    top = names[np.argsort(clf.coef_[0])[-12:]][::-1]
    print(f"bag-of-words test AUC (pair-held-out): {auc:.3f}")
    print("top lonely-indicative n-grams:", ", ".join(top))
    print(
        "If these are just the planted cue words, consider whether the probe "
        "could be reading anything more than lexical content."
    )


if __name__ == "__main__":
    main()
