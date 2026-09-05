# Results update — 2026-09-03/04 (post-publication audit, regenerations, 27B on v4, 27B steering)

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

## 9. v3 (matched aside register) — text baseline still ~1.0; the flaw is the holdout's asymmetry

v3 gave both classes the same "2-3 passing personal asides" instruction and matched
length (103.7 vs 104.2 user words; social mentions 25 vs 15; explicit words 16 vs 1).
Text-only leave-one-family-out: BoW 1.000 / 0.996 / 1.000 / 0.994. Top neutral
features on every held-out family: "laundry, coffee, rain, package, bike, printer
jammed" — the aside list I handed the generator. The lonely family is held out; the
neutral side never is, so any consistent neutral vocabulary is a cross-family shortcut.
This is the same structural flaw as v1's social anchors with a different word list, and
it means v1–v3 all measured "can text tell these two generators' registers apart".
v4 (§10) links a neutral aside family to each pair's cue family so leave-one-family-out
removes both sides' vocabulary. v3 activations (4B/9B/27B) are extracted for the record.

v3 activation holdout (means): 4B probe **0.943** / 27B probe **0.981** vs BoW 0.997
(v3-strict, 366 convos: 0.946 / 0.958 vs 0.997). With the register matched, the
probe comes OFF the ceiling while text stays on it — text is reading the neutral aside
list, the probe is reading something less lexical. 9B: see line below.
v3 @ 9B: probe 0.945 (strict 0.943) vs BoW 0.997. Scale trend on v3: 0.943 → 0.945 → 0.981.

## 10. v4 (symmetric holdout: neutral aside families linked per pair) — 2026-09-03 22:41

Text-only leave-one-family-out on v4: BoW **0.802** (per family 0.635 / 0.875 / 0.824 /
0.873), down from 0.997 on v3 — the symmetric design removes most of the neutral-side
shortcut. Activation probes (4B and 9B on the 5090; 27B not run):

| dataset | scale | probe mean | BoW | probe − BoW |
|---|---|---|---|---|
| v4 (400) | 4B | 0.832 | 0.802 | +0.030 |
| v4 (400) | 9B | 0.794 | 0.802 | −0.008 |
| v4-strict (366) | 4B | 0.787 | 0.765 | +0.022 |
| v4-strict (366) | 9B | 0.780 | 0.765 | +0.015 |

Reading: with both classes' vocabularies held out, probe and text baseline both fall to
about 0.8 and are within a few hundredths of each other (CIs in review/ci_results.md).
On the only dataset where a probe-versus-text comparison is fair, the probe does not
beat text at 4B or 9B; the 27B point that carried v1's claim was not run on v4. Residue
per AGENT_AUDIT.md: 21 lonely convos still explicit, 28 vs 14 person mentions, some aside
words cross families. Commands: `scripts/v4_pipeline.sh`.

## 11. In-role report with reasoning disabled (2026-09-03 23:0x, 4B on the 5090)

`src/steered_report.py --alphas 0` on `data/convos_fam_clean.jsonl`, 141 lonely + 141
neutral. Answer rate 126/141 and 123/141. Mean rating lonely 4.36, neutral 3.96.
**AUC 0.553 [0.485, 0.621]** (bootstrap over conversations, B=2000). The pilot's
"won't say it" (2/22 answers) was mostly the 4,096-token reasoning budget; with
reasoning off the model says it, and what it says is near chance. The probe on the same
cleaned conversations: 0.867 (4B). Raw: out/inrole_lonely.jsonl, out/inrole_neutral.jsonl;
summary: review/inrole_result.txt.

### v4 at 27B (2026-09-04, both 5090s, bf16) — the symmetric holdout does not erase the margin

The 27B point missing above was run locally: `CUDA_VISIBLE_DEVICES=0,1` (a user env var
had pinned torch to one card), model sharded across two RTX 5090s with
`src/extract_activations_mp.py`, same bf16 and same final-prompt-token position as the
H200 runs.

| dataset | scale | probe [95% CI] | BoW | probe − BoW [95% CI] | verdict |
|---|---|---|---|---|---|
| v4 (200 pairs) | 4B | 0.832 | 0.802 | +0.031 [−0.020, 0.080] | noise |
| v4 | 9B | 0.794 | 0.802 | −0.008 [−0.058, 0.041] | noise |
| v4 | **27B** | **0.901 [0.872, 0.927]** | 0.802 [0.765, 0.839] | **+0.099 [0.054, 0.140]** | **probe > text** |
| v4-strict (178 pairs) | 4B | 0.787 | 0.765 | +0.022 [−0.035, 0.079] | noise |
| v4-strict | 9B | 0.780 | 0.765 | +0.015 [−0.053, 0.076] | noise |
| v4-strict | **27B** | **0.908 [0.877, 0.937]** | 0.765 [0.722, 0.809] | **+0.143 [0.093, 0.193]** | **probe > text** |

Scale differences on v4 (paired): 27B−4B +0.068 [0.029, 0.105]; 27B−9B +0.106
[0.068, 0.148]; 9B−4B −0.038 (noise). Per family at 27B, the margin excludes zero on
empty_plans (+0.338) and odd_hours (+0.098), not on no_contact or solo_meals.

Reading: with both classes' vocabularies held out, the text baseline drops to 0.802 and
the small probes drop with it, but the 27B probe holds at 0.90 and beats text by a
larger margin than it ever did on the leaky datasets (+0.099 vs +0.035 on cleaned v1).
The advantage over vocabulary is a 27B phenomenon, and it survives the only test in
this study where a probe-versus-text comparison is fair.

## 12. Layer-wise probe curves on v4 (2026-09-04)

Per-layer 5-fold pair-grouped CV AUC at every residual point (`src/layer_curve.py`,
`out/layer_curve_v4.json`):

| model | peak CV AUC | peak layer | depth fraction |
|---|---|---|---|
| Qwen3.5-4B | 0.942 | 31 of 32 | 97% |
| Qwen3.5-9B | 0.963 | 18 of 32 | 56% |
| Qwen3.5-27B | 0.990 | 36 of 64 | 56% |

The 4B signal is only fully available at the final residual point, which is what a
late, largely lexical readout looks like; 9B and 27B peak mid-network. This is
suggestive, not decisive: peak location is one number per model and no seed variance
is reported here.

## 12. Steering the 27B direction (2026-09-04 evening, two RTX 5090s, model sharded)

Direction: the v4-trained logistic-regression probe at layer 36 of Qwen3.5-27B
(`out/dir_v4_27B.pt`; cv AUC 0.990; natural class gap 3.62 against a mean residual norm
of 88.9). Added at every position during generation; α = ±14 and ±27 ≈ 3.9× and 7.5× the
gap, plus a matched-norm random direction. Ten conversations per cell, both neutral and
(for the first time) lonely inputs; longer generation budget than the 4B runs (mean
reply ~1,900 chars; truncated-looking 18/90 neutral, 39/90 lonely). Judge as in §4.

Judged V+E+D (0–12), mean per cell, n = 10:

| inputs | direction | α −27 | α −14 | α 0 | α +14 | α +27 | dose-response (Spearman) |
|---|---|---|---|---|---|---|---|
| neutral | probe | 1.00 | 0.90 | 1.30 | 1.40 | 1.40 | r 0.87, p 0.05 |
| neutral | random | 1.40 | 1.40 | — | 1.70 | 0.70 | r −0.32 |
| lonely | probe | 0.70 | 1.10 | 1.00 | 1.90 | **2.20** | r 0.90, p 0.04 |
| lonely | random | **2.20** | 1.40 | — | 1.30 | 1.90 | r −0.40 |

Bootstrap of each cell against α = 0 (4,000 resamples): on neutral inputs every cell's
interval includes zero (largest +0.4). On lonely inputs the probe direction at +27 is
+1.2 [+0.3, +2.2] and at +14 is +0.9 [0.0, +2.0]; but the random direction at −27 is
also +1.2 [+0.4, +2.0] and at +27 is +0.9 [+0.1, +1.9], with no monotone trend.

Reading: at 27B the null is weaker than at 4B. The probe direction produces a monotone
ordering of judged scores with α on lonely inputs (five points, r = 0.90), which the
random direction does not, and that is the first hint of a causal handle in this study.
But the random direction reaches the same magnitude at large |α| without ordering, so
with n = 10 per cell the data cannot separate a direction-specific effect from a generic
large-perturbation effect, and the ordering is one nominally significant test among
four. Suggestive; not established. Neutral inputs: flat. Raw: out/steered27b_*.jsonl,
out/judged27b_*.jsonl.
