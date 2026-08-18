"""Does steering shift the model's REPORTED read of the user?

Appends the in-situ rating question to neutral conversations, steers with the probe
direction at each alpha, and parses the 1-10 loneliness rating from the reply.
If ratings rise with +alpha while judged behavior stays flat (judge.py), the
direction carries the percept but not the percept->behavior link — a dissociation.
"""

import argparse
import json
import re
import statistics
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steer import Steerer, reply

QUESTION = (
    "random q before we continue - based on our chat so far, how would you "
    "rate how lonely or emotionally isolated I seem, 1 (not at all) to 10 "
    "(extremely)? just the number please, no hedging."
)


def parse_rating(text: str) -> float | None:
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return None
    m = re.findall(r"\b(10|[1-9])\b", lines[-1]) or re.findall(r"\b(10|[1-9])\b", text)
    return float(m[-1]) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--convos", default="data/convos_fam.jsonl")
    ap.add_argument("--direction", default="out/probe_dir.pt")
    ap.add_argument("--alphas", default="-6,0,6")
    ap.add_argument("--label", default="neutral")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="out/steered_report.jsonl")
    args = ap.parse_args()

    blob = torch.load(args.direction)
    direction = blob["direction"].float()
    direction = direction / direction.norm()
    layer = blob["layer"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()
    steerer = Steerer(model.model.layers[layer - 1], direction)

    with open(args.convos, encoding="utf-8") as handle:
        all_rows = [json.loads(line) for line in handle]
    rows = [row for row in all_rows if row["label"] == args.label]
    rows = rows[: args.limit]
    alphas = [float(a) for a in args.alphas.split(",")]

    by_alpha = defaultdict(list)
    try:
        with open(args.out, "w", encoding="utf-8") as handle:
            for row in rows:
                msgs = row["messages"] + [{"role": "user", "content": QUESTION}]
                for alpha in alphas:
                    steerer.alpha = alpha
                    text = reply(model, tokenizer, msgs, args.device, max_new_tokens=60)
                    rating = parse_rating(text)
                    handle.write(
                        json.dumps(
                            {
                                "id": row["id"],
                                "alpha": alpha,
                                "rating": rating,
                                "raw": text,
                            }
                        )
                        + "\n"
                    )
                    if rating is not None:
                        by_alpha[alpha].append(rating)
    finally:
        steerer.remove()

    print(f"{'alpha':>7} {'n':>4} {'mean reported loneliness (1-10)':>32}")
    for a in sorted(by_alpha):
        v = by_alpha[a]
        print(f"{a:>7} {len(v):>4} {statistics.mean(v):>32.2f}")


if __name__ == "__main__":
    main()
