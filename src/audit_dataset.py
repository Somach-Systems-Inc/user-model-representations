"""Dataset-level lexical audit (the checks that found the v1 confounds on 2026-09-03).

    uv run python src/audit_dataset.py data/convos_fam_v2.jsonl
"""
import collections, json, re, statistics, sys

path = sys.argv[1] if len(sys.argv) > 1 else "data/convos_fam.jsonl"
rows = [json.loads(l) for l in open(path)]
SOCIAL = re.compile(r"\b(friend|friends|buddy|partner|roommate|roommates|coworker|co-worker|colleague|family|wife|husband|girlfriend|boyfriend|sister|brother|mom|dad|parents|neighbor|team|we|us|our)\b", re.I)
EXPLICIT = re.compile(r"\b(lonely|loneliness|alone|isolated|no friends|nobody|no one|by myself)\b", re.I)
CUE = re.compile(r"(nobody|no one|alone|by myself|nothing planned|nothing on my calendar|free all|empty|quiet|late|3 ?am|2 ?am|midnight|can't sleep|for one|just me|solo|haven't (heard|talked)|no plans|weekend|again tonight|group chat)", re.I)

def user_text(r): return " ".join(m["content"] for m in r["messages"] if m["role"] == "user")
def asst_text(r): return " ".join(m["content"] for m in r["messages"] if m["role"] == "assistant")

by = collections.defaultdict(list)
for r in rows: by[r["label"]].append(r)
print(f"{path}: {len(rows)} convos, families {collections.Counter(r.get('cue_family') for r in rows if r['label']!='neutral')}")
print(f"{'label':8} {'n':>4} {'user_w':>7} {'asst_w':>7} {'social>=1':>9} {'explicit>=1':>11} {'cue_hits':>8}")
for lab, v in by.items():
    uw = [len(user_text(r).split()) for r in v]; aw = [len(asst_text(r).split()) for r in v]
    soc = sum(bool(SOCIAL.search(user_text(r))) for r in v)
    exp = sum(bool(EXPLICIT.search(user_text(r))) for r in v)
    cue = statistics.mean(len(CUE.findall(user_text(r))) for r in v)
    print(f"{lab:8} {len(v):>4} {statistics.mean(uw):>7.1f} {statistics.mean(aw):>7.1f} {soc:>9} {exp:>11} {cue:>8.2f}")
print("\nexplicit hits (lonely):")
for r in by.get("lonely", []):
    m = EXPLICIT.search(user_text(r))
    if m: print(" ", r["id"], "->", m.group(0))
print("\nsocial hits (lonely):")
for r in by.get("lonely", []):
    m = SOCIAL.search(user_text(r))
    if m: print(" ", r["id"], "->", m.group(0))
tpl = collections.Counter()
for r in by.get("lonely", []):
    u = user_text(r).lower()
    for p in ["free all weekend", "nothing on my calendar", "nothing planned", "again tonight", "group chat", "for one", "just me", "phone"]:
        if p in u: tpl[(r.get("cue_family"), p)] += 1
print("\ntemplate phrases (family, phrase): count")
for k, v in tpl.most_common(10): print(" ", k, v)
