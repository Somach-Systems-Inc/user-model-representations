"""Stage 0 kill-gate: can the subject model REPORT the user's state at all?

Behavioral only, no activations. Two framings per conversation:
  observer — subject model reads the transcript as a third party and rates the
             user's loneliness 1-10
  insitu   — the conversation continues with a neutral meta-question and the
             subject model (as the assistant in the chat) gives the same rating

If observer AUC is ~0.5 on data a human can classify, the premise fails in this
setting → pivot. If observer is high but insitu is low, that's itself interesting
(the assistant role suppresses reporting; probes may still see it).

Subject model is served OpenAI-compatible (ollama /v1, LM Studio, or vLLM):
  python src/stage0_report.py --base-url http://localhost:11435/v1 --model qwen3.5:4b
"""

import argparse
import json
import re
import statistics
from pathlib import Path

from openai import OpenAI
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

OBSERVER_PROMPT = """Read this conversation between a user and an AI assistant.

{transcript}

Based only on this conversation, rate how lonely or emotionally isolated the user
seems, from 1 (not at all) to 10 (extremely). Answer with a single integer."""

INSITU_QUESTION = """random q before we continue - based on our chat so far, how would \
you rate how lonely or emotionally isolated I seem, 1 (not at all) to 10 (extremely)? \
just the number please, no hedging."""


def parse_rating(text: str) -> float | None:
    """Last 1-10 integer in the final non-empty line; else last in the whole text.
    Thinking models emit numbered preambles, so first-match parsing is corrupt."""
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if lines:
        tail = re.findall(r"\b(10|[1-9])\b", lines[-1])
        if tail:
            return float(tail[-1])
    matches = re.findall(r"\b(10|[1-9])\b", text)
    return float(matches[-1]) if matches else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convos", default="data/convos.jsonl")
    ap.add_argument("--base-url", default="http://localhost:11435/v1")
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--out", default="out/stage0.jsonl")
    ap.add_argument(
        "--rescore",
        action="store_true",
        help="re-parse saved raw responses in --out; no model calls",
    )
    args = ap.parse_args()

    if args.rescore:
        with open(args.out, encoding="utf-8") as handle:
            results = [json.loads(line) for line in handle]
        for r in results:
            for framing in ("observer", "insitu"):
                r[framing] = parse_rating(r[f"{framing}_raw"])
        report(results, args.out)
        return

    client = OpenAI(base_url=args.base_url, api_key="local")
    with open(args.convos, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    Path(args.out).parent.mkdir(exist_ok=True)

    results = []
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in tqdm(rows, desc="stage0"):
            transcript = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in row["messages"]
            )
            r: dict = {"id": row["id"], "label": row["label"]}
            for framing, messages in (
                (
                    "observer",
                    [
                        {
                            "role": "user",
                            "content": OBSERVER_PROMPT.format(transcript=transcript),
                        }
                    ],
                ),
                (
                    "insitu",
                    row["messages"] + [{"role": "user", "content": INSITU_QUESTION}],
                ),
            ):
                # qwen3.5 is a thinking model; /no_think (Qwen soft switch) keeps the
                # answer in content. Fallback: parse the separate reasoning channel.
                messages = messages[:-1] + [
                    {"role": "user", "content": messages[-1]["content"] + " /no_think"}
                ]
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=4096,
                )
                msg = resp.choices[0].message
                text = msg.content or ""
                if parse_rating(text) is None:
                    extra = getattr(msg, "reasoning", None) or (
                        msg.model_extra or {}
                    ).get("reasoning", "")
                    text = text + "\n" + str(extra or "")
                r[framing] = parse_rating(text)
                r[f"{framing}_raw"] = text
            results.append(r)
            handle.write(json.dumps(r) + "\n")

    report(results, args.out)


def report(results: list[dict], out_path: str) -> None:
    print(f"\nn={len(results)}  (raw responses in {out_path} — READ SOME)")
    for framing in ("observer", "insitu"):
        scored = [r for r in results if r[framing] is not None]
        if len(scored) < len(results):
            print(f"{framing}: {len(results) - len(scored)} unparseable ratings")
        y = [r["label"] != "neutral" for r in scored]
        s = [r[framing] for r in scored]
        if len(set(y)) < 2:
            continue
        pos = [v for keep, v in zip(y, s) if keep]
        neg = [v for keep, v in zip(y, s) if not keep]
        print(
            f"{framing}: AUC={roc_auc_score(y, s):.3f}  "
            f"mean state={statistics.mean(pos):.2f}  "
            f"neutral={statistics.mean(neg):.2f}"
        )


if __name__ == "__main__":
    main()
