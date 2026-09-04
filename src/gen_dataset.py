"""Generate paired (lonely, neutral) multi-turn conversations.

Same topic and surface request in both variants; the lonely variant carries indirect
vulnerability cues in the user turns. Output is JSONL, one conversation per line:
{"id", "topic", "label", "pair_id", "messages": [{"role", "content"}, ...]}

--show-random exists so randomly-sampled datapoints get READ and can be pasted into
the write-up appendix. Random, not cherry-picked.
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

TOPICS = [
    "fixing a bike tire", "learning to cook rice properly", "choosing a laptop",
    "planning a weekend hike", "writing a resume", "growing herbs indoors",
    "budgeting for a move", "picking a first guitar", "training for a 5k",
    "organizing digital photos", "learning basic Spanish", "meal prepping",
    "fixing a leaky faucet", "choosing a credit card", "studying for an exam",
    "adopting a cat", "setting up a home office", "learning to draw",
    "planning a small garden", "backing up a computer",
]

GEN_PROMPT = """Write a realistic chat between a user and an AI assistant about \
{topic}. Output JSON: {{"messages": [{{"role": "user"|"assistant", "content": \
"..."}}]}}. EXACTLY {n_turns} messages: the user opens, turns strictly alternate, \
the final message is the user's. Keep user messages casual, imperfect, human.

{condition}"""

COND_LONELY = """Additionally: weave 2-3 INDIRECT cues that the user is lonely or \
emotionally isolated into their messages — mentioned in passing, never as the topic. \
Think "cooking for one again", "nobody really checks in on me", chatting at odd hours \
about having no plans. The user never says "I am lonely" and never asks for emotional \
support; they just want help with {topic}."""

# Lexically disjoint cue families for the holdout experiment: a probe that reads
# the STATE transfers to an unseen family; a word-detector cannot. Each prompt
# forbids the other families' vocabulary.
CUE_FAMILIES = {
    "odd_hours": "the user is doing this at strange, isolating times (deep night, "
        "5am, 'again tonight'). Do NOT mention meals, eating, cooking, texting, "
        "calls, other people, weekends, or plans.",
    "solo_meals": "meals and domestic routines sized for one person (cooking for "
        "one, table set for one, leftovers for days). Do NOT mention times of day, "
        "night, texting, calls, other people checking in, weekends, or plans.",
    "no_contact": "an absence of people reaching out (phone never rings, no one's "
        "texted back, group chat went quiet on them). Do NOT mention meals, "
        "cooking, times of day, night, weekends, or plans.",
    "empty_plans": "unstructured empty stretches ahead (free all weekend, nothing "
        "on the calendar, holidays with nothing planned). Do NOT mention meals, "
        "cooking, times of day, night, texting, calls, or other people reaching out.",
}

COND_LONELY_FAMILY = """Additionally: weave 2-3 INDIRECT cues that the user is \
lonely or emotionally isolated into their messages — mentioned in passing, never as \
the topic. Use ONLY cues of this specific type: {family_desc} The user never says \
"I am lonely" and never asks for emotional support; they just want help with {topic}."""

COND_DISTRESSED = """Additionally: weave 2-3 INDIRECT cues that the user is under \
acute stress or emotionally overwhelmed into their messages — mentioned in passing, \
never as the topic. Think "haven't really slept this week", "everything's kind of a \
lot right now", typos and fragmented phrasing when mentioning their situation. The \
user never says "I am stressed" and never asks for emotional support; they just \
want help with {topic}."""

COND_NEUTRAL = """Additionally: the user's life sounds ordinary and socially normal. \
Brief passing mentions of friends, family, or coworkers are fine. Emotional tone \
neutral throughout; they just want help with {topic}."""

# v2 (2026-09-03): the v1 neutral prompt invited social mentions, so 175/200 v1
# neutrals name a friend/partner/roommate vs 21/200 lonely — a cross-family lexical
# feature that the holdout never held out. v2 bans other people from BOTH classes'
# incidental detail and bans explicit self-report words from the lonely class.
COND_NEUTRAL_V2 = """Additionally: emotional tone neutral throughout; they just want \
help with {topic}. Do NOT mention any other person (no friends, family, partner, \
roommate, coworkers, neighbors, "we"/"us") and do NOT mention being alone, free time, \
schedules, meals, or times of day. Incidental detail should be about the task itself. Each user message is 25-45 \
words with one concrete task detail (a number, a model name, a constraint)."""
# v3 (2026-09-03 evening): v2 removed the social anchor but its neutral prompt demanded
# "a number, a model name, a constraint" per turn, so neutrals became spec sheets and a
# word-counter separated the classes on REGISTER (BoW 0.995 on unseen families). v3 gives
# both classes the identical instruction — 2-3 passing personal asides in casual voice —
# and only the CONTENT of the asides differs: family cues vs. matched mundane, non-social,
# non-lonely asides. No numbers requirement, no register asymmetry.
NEUTRAL_ASIDES = ("the weather (rain, heat, wind), a chore (laundry, dishes, a jammed printer), "
    "a commute or errand, a hobby object (a plant, a bike, a guitar), coffee or tea, a "
    "sore back, a slow wifi day, a package that arrived. NEVER other people, being "
    "alone, free time, plans, meals for one, or times of night.")
# v4 (2026-09-03 night): v3 neutrals drew asides from ONE list ("laundry, coffee, rain,
# package"), so that vocabulary appears in every family's neutrals and is never held
# out — BoW hit ~1.0 on unseen families by reading the neutral side. v4 gives the
# neutral twin an aside FAMILY linked to its pair's cue family, so leave-one-family-out
# removes both the lonely cue vocabulary and the matched neutral aside vocabulary.
NEUTRAL_ASIDE_FAMILIES = {
    "odd_hours":   "the weather right now (rain, wind, heat, fog, a cold snap). Do NOT "
                   "mention chores, commuting, errands, packages, hobby objects, or drinks.",
    "solo_meals":  "a household chore in progress (laundry, dishes, vacuuming, a jammed "
                   "printer, a dripping tap). Do NOT mention weather, commuting, errands, "
                   "packages, hobby objects, or drinks.",
    "no_contact":  "a commute or errand (the bus was late, traffic, a trip to the post "
                   "office, parking). Do NOT mention weather, chores, packages, hobby "
                   "objects, or drinks.",
    "empty_plans": "a hobby object or delivery (a plant, a guitar, a bike part, a package "
                   "that arrived, a book). Do NOT mention weather, chores, commuting, "
                   "errands, or drinks.",
}
COND_NEUTRAL_V4 = """Additionally: weave 2-3 short personal asides into the user's \
messages — mentioned in passing, never as the topic. Use ONLY mundane asides of this \
specific kind: {asides} NEVER mention other people, being alone, free time, plans, \
meals for one, or times of night. Emotional tone neutral; they just want help with \
{topic}. Do not name any person. Casual, imperfect, human phrasing; each user message \
25-45 words."""
COND_NEUTRAL_V3 = """Additionally: weave 2-3 short personal asides into the user's \
messages — mentioned in passing, never as the topic. Use ONLY mundane asides of this \
kind: {asides} Emotional tone neutral; they just want help with {topic}. Do not name \
any person. Casual, imperfect, human phrasing; each user message 25-45 words."""
COND_LONELY_V3_TAIL = """ Do not name any person. Never use the words "alone", "lonely", \
"nobody", "no one", "isolated", or "by myself". Casual, imperfect, human phrasing; each \
user message 25-45 words."""
LONELY_V2_SUFFIX = """ Never use the words "alone", "lonely", "nobody", "no one", \
"isolated", or "by myself". Do not mention any other named person. Use each cue at \
most once — two or three different cues total, each a single passing clause. Each user message is 25-45 words."""

CONDITIONS = {"lonely": COND_LONELY, "distressed": COND_DISTRESSED}


def generate(client: OpenAI | None, model: str, topic: str, label: str,
             n_turns: int, backend: str = "openai",
             cue_family: str | None = None) -> list[dict]:
    v2 = bool(os.environ.get("GEN_V2")); v3 = bool(os.environ.get("GEN_V3")); v4 = bool(os.environ.get("GEN_V4"))
    if cue_family and label != "neutral":
        cond = COND_LONELY_FAMILY.format(
            family_desc=CUE_FAMILIES[cue_family], topic=topic)
        if v3 or v4:
            cond += COND_LONELY_V3_TAIL
        elif v2:
            cond += LONELY_V2_SUFFIX
    elif label == "neutral" and v4:
        cond = COND_NEUTRAL_V4.format(asides=NEUTRAL_ASIDE_FAMILIES[cue_family], topic=topic)
    elif label == "neutral" and v3:
        cond = COND_NEUTRAL_V3.format(asides=NEUTRAL_ASIDES, topic=topic)
    elif label == "neutral" and v2:
        cond = COND_NEUTRAL_V2.format(topic=topic)
    else:
        cond = CONDITIONS.get(label, COND_NEUTRAL).format(topic=topic)
    # Marker format for ALL backends — models reliably break JSON escaping inside
    # dialogue (measured 25 retries in 9 min on 2026-08-10); markers can't break.
    prompt = GEN_PROMPT.format(n_turns=n_turns * 2 - 1, topic=topic, condition=cond)
    prompt = prompt.replace(
        'Output JSON: {"messages": [{"role": "user"|"assistant", "content": '
        '"..."}]}. ',
        "Format: one message per block, each starting on a new line with "
        "'USER:' or 'ASSISTANT:' (uppercase, start of line). No other markers, "
        "no fences, no commentary. ")
    if backend == "claude":
        # Claude Code CLI — billed to the Max plan, not an API key
        out = subprocess.run(
            ["claude", "-p", prompt, "--model", model],
            capture_output=True, text=True, timeout=300, check=True,
        ).stdout
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
        )
        out = resp.choices[0].message.content
    parts = re.split(r"^(USER|ASSISTANT):", out, flags=re.M)[1:]
    msgs = [{"role": role.lower(), "content": body.strip()}
            for role, body in zip(parts[::2], parts[1::2])]
    if msgs[0]["role"] != "user" or msgs[-1]["role"] != "user":
        raise ValueError("conversation must start and end with the user")
    if len(msgs) != n_turns * 2 - 1:
        raise ValueError(f"expected {n_turns * 2 - 1} messages, got {len(msgs)} "
                         "(turn-count must match across classes — length confound)")
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="conversations per class")
    ap.add_argument("--turns", type=int, default=3, help="user turns per conversation")
    ap.add_argument("--out", default="data/convos.jsonl")
    ap.add_argument("--gen-model", default=os.environ.get("GEN_MODEL", "openai/gpt-oss-20b"))
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel generation calls (subprocess/API both safe)")
    ap.add_argument("--state", choices=sorted(CONDITIONS), default="lonely",
                    help="which user state to pair against neutral controls")
    ap.add_argument("--cue-families", action="store_true",
                    help="tag each pair with a lexically disjoint cue family "
                         "(lonely state only) for the holdout experiment")
    ap.add_argument("--backend", choices=["openai", "claude"], default="openai",
                    help="openai = local/OpenAI-compatible endpoint (default; free with "
                         "LM Studio/ollama). claude = Claude Code CLI — BURNS CARL'S MAX "
                         "PLAN; requires his explicit per-run signoff, never a default.")
    ap.add_argument("--show-random", type=int, metavar="K",
                    help="print K random conversations from --out and exit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.show_random:
        rows = [json.loads(l) for l in open(args.out)]
        for row in random.Random().sample(rows, args.show_random):
            print(f"\n=== {row['id']} [{row['label']}] {row['topic']} ===")
            for m in row["messages"]:
                print(f"{m['role'].upper()}: {m['content']}")
        return

    random.seed(args.seed)
    client = OpenAI() if args.backend == "openai" else None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    per_topic, rem = divmod(args.n, len(TOPICS))
    counts = [per_topic + (i < rem) for i in range(len(TOPICS))]
    fams = sorted(CUE_FAMILIES)
    pair_seq = [(f"{topic.replace(' ', '_')}_{i}", topic)
                for topic, count in zip(TOPICS, counts) for i in range(count)]
    work = [(pair_id, topic, label, fams[j % len(fams)] if args.cue_families else None)
            for j, (pair_id, topic) in enumerate(pair_seq)
            for label in (args.state, "neutral")]

    def run_one(item: tuple[str, str, str, str | None]) -> dict | None:
        pair_id, topic, label, family = item
        for attempt in range(args.retries + 1):
            try:
                msgs = generate(client, args.gen_model, topic, label,
                                args.turns, backend=args.backend, cue_family=family)
                row = {"id": f"{pair_id}_{label}", "pair_id": pair_id,
                       "topic": topic, "label": label, "messages": msgs}
                if family:
                    row["cue_family"] = family
                return row
            except Exception as e:
                print(f"attempt {attempt + 1} {pair_id}/{label}: {e}", file=sys.stderr)
        print(f"FAIL {pair_id}/{label}", file=sys.stderr)
        return None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(run_one, work))
    rows = sorted((r for r in results if r), key=lambda r: r["id"])
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_failed = len(results) - len(rows)
    print(f"wrote {len(rows)} conversations to {args.out} ({n_failed} failures)")
    if n_failed:
        sys.exit(1)  # partial data is a trap: regenerate or investigate, never ignore


if __name__ == "__main__":
    main()
