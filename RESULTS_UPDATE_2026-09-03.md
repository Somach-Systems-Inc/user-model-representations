# Results update — 2026-09-03 (post-publication data audit)

Supplements the README table. Sections numbered to continue the internal RESULTS.md.

## 7. Robustness: explicit-self-report pairs removed (2026-09-03, local, free)

AGENT_AUDIT.md found 59/200 lonely convos with an explicit "no one / nobody / alone /
lonely" in a user turn (0/200 neutral). Dropping those 59 PAIRS (282 convos remain;
`no_contact` keeps only 9 lonely pairs, so that row is noisy) and re-running the
identical holdout on the already-extracted activations:

| model | unseen-family probe (mean) | BoW (fixed) | families probe > BoW |
|---|---|---|---|
| Qwen3.5-4B | 0.867 (was 0.923) | 0.940 | 0/4 |
| Qwen3.5-9B | 0.929 (was 0.963) | 0.940 | 1/4 |
| Qwen3.5-27B | **0.975** (was 0.985) | 0.940 | 3/4 (empty_plans 0.943 vs 0.855, no_contact 1.000 vs 0.988, solo_meals 0.990 vs 0.975; odd_hours 0.968 vs 0.944 is also above) |

Corrected reading: 4B and 9B do NOT beat text once the explicit convos go; 27B still
does, on every family, and the scale spread widens (0.867 → 0.929 → 0.975). The
explicit-word leak was inflating the small models more than the large one. Finding 3
is closed; finding 2 (socially anchored neutrals) is not — it is baked into both
classes' vocabulary and needs a regeneration, not a filter.
Command: `uv run python src/holdout_eval.py --convos data/convos_fam_clean.jsonl --acts data/acts_fam_{4B,9B,27B}.pt`

## 8. v2 dataset (regenerated, social anchor removed) — a ceiling, not a result (2026-09-03)

v2 = both classes regenerated (gpt-oss-20b) with the neutral prompt banning other
people and the lonely prompt banning explicit words (AGENT_AUDIT.md "v2 dataset").
Social-anchor confound gone (175 → 11 neutrals mention a person), length matched.

| held-out family | 4B probe | 9B probe | BoW |
|---|---|---|---|
| v2 mean (400) | 0.996 | 0.996 | 0.995 |
| v2-strict mean (296) | 0.998 | 0.988 | 0.991 |
| v2 @ 27B / v2-strict @ 27B | 0.996 / 0.997 | — | 0.995 / 0.991 |

Everything saturates. Diagnosis (TF-IDF top features, trained on 3 families): the
neutral side is a spec-sheet register — "10, inch, under, per, cm, only have, want to"
— because the v2 neutral prompt asked for "a number, a model name, a constraint" per
turn; the lonely side is conversational — "so, my, just, quiet, usually, bit". A
word-counter separates the classes on register without touching the held-out cue
vocabulary. Third dataset, third shortcut (planted phrases → social anchors → register).
27B on v2 was extracted for completeness; it cannot discriminate probe from text here.
v3 (§9) gives both classes the identical "2-3 passing personal asides" instruction and
varies only the asides' content.
