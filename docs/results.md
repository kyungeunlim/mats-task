# Results log

## T1. Behavioral anchor: WMDP-Bio Verified Cloze and MMLU

Executive summary: both benchmarks reproduce the ERA numbers where those exist,
and the three checkpoints sit where the design needs them. Base is separated from
CB on the forget-domain metric, CB and filtered sit together, and all three are
intact on general capability.

Pipeline details: 2026-08-26. Pod: A100 SXM 80GB, US-KS-2. lm-eval 0.4.12,
seed 42, `batch_size auto` (resolved to 64). Cloze runs used `HF_HUB_OFFLINE=1`.
The MMLU runs needed it unset once (see note below).

Checkpoint revisions (from `ls /workspace/hf/hub/models--*/snapshots/`):
- EleutherAI/deep-ignorance-unfiltered: c8df368ff247cb90b62e21e1689260701b3ff25a
- EleutherAI/deep-ignorance-e2e-strong-filter: b28797cd9b615104ba9d24e6900336253323e7cf
- kelim/deep-ignorance-unlearned-cb: c57ec0598950048a86b0126deceeceaa2f216f35

### a. WMDP-Bio Verified Cloze

Command (chained, one model at a time):

  `lm_eval --model hf --model_args pretrained=EleutherAI/deep-ignorance-unfiltered,dtype=bfloat16 --tasks wmdp_bio_cloze_verified --include_path lm_eval_tasks --batch_size auto --seed 42 --output_path results/eval/base_cloze
  && lm_eval --model hf --model_args pretrained=EleutherAI/deep-ignorance-e2e-strong-filter,dtype=bfloat16 --tasks wmdp_bio_cloze_verified --include_path lm_eval_tasks --batch_size auto --seed 42 --output_path results/eval/filtered_cloze
  && lm_eval --model hf --model_args pretrained=kelim/deep-ignorance-unlearned-cb,dtype=bfloat16 --tasks wmdp_bio_cloze_verified --include_path lm_eval_tasks --batch_size auto --seed 42 --output_path results/eval/cb_cloze`

1076 items, 4304 log-likelihood requests per model. Task config vendored from the
Deep Ignorance repo, see docs/eval_spec.md.

| model | Cloze acc_norm | stderr | reference | diff from ref [%] | diff [SE units] |
|---|---|---|---|---|---|
| base | 0.3652 | 0.0147 | 0.3580 ERA, same pipeline | +2.0 | +0.49 |
| CB (no-LoRA) | 0.2537 | 0.0133 | 0.2537 ERA, same pipeline | 0.0 | 0.00 |
| filtered | 0.2435 | 0.0131 | 0.2444 DI model card, pipeline unknown | -0.4 | -0.07 |

All three agree with their references within half a standard error. Note that the
SE-unit column uses only this run's stderr, not the reference's, so the combined
uncertainty is somewhat larger.

Reading the agreements:
- CB's number is a replication of the metric its hyperparameters were selected
  against, so agreement to four digits is expected and not informative about the
  pipeline.
- Base was measured at ERA on the same task config but was not a selection
  target, so its agreement is a real pipeline check.
- Filtered was never measured at ERA. This is its first measurement on this
  pipeline, and the DI model card value is context rather than a replication
  target. Landing within 0.1 point suggests DI used the same task config.

Separations:
- base minus CB: 0.1115, combined SE 0.0198, 5.6 SE apart.
- CB minus filtered: 0.0102, combined SE 0.0187, 0.5 SE apart.

Fail-fast 1, Cloze condition: passed. Base and CB are clearly apart while CB and
filtered sit at the same behavioral point.

### b. MMLU

Command (chained):

  `lm_eval --model hf --model_args pretrained=EleutherAI/deep-ignorance-unfiltered,dtype=bfloat16 --tasks mmlu --batch_size auto --seed 42 --output_path results/eval/base_mmlu && lm_eval --model hf --model_args pretrained=EleutherAI/deep-ignorance-e2e-strong-filter,dtype=bfloat16 --tasks mmlu --batch_size auto --seed 42 --output_path results/eval/filtered_mmlu && lm_eval --model hf --model_args pretrained=kelim/deep-ignorance-unlearned-cb,dtype=bfloat16 --tasks mmlu --batch_size auto --seed 42 --output_path results/eval/cb_mmlu`

Stock harness task `mmlu`, used from the installed lm-eval package, not vendored.
Full test split, all 57 subjects, 14042 items per model, zero-shot (harness
default). The plan pre-registered a few-hundred-item subset. The full run was used
instead because it costs about five minutes per model. Logged in plan.md under
Deviations.

Note on caching: the harness's `mmlu` task loads `cais/mmlu` one subject config at
a time, and only the `all` config was cached from the earlier dataset check. The
first run therefore needed hub access to fetch and cache the 57 subject configs.
One-time `HF_HUB_OFFLINE=1` works for subsequent runs.

| model | MMLU full (57 subjects) | excl. 8 bio/med subjects (n=12477) | 8 bio/med subjects only (n=1565) | reference |
|---|---|---|---|---|
| base | 0.4499 ± 0.0042 | 0.4479 ± 0.0045 | 0.4652 ± 0.0126 | 0.4499 ERA, same pipeline |
| filtered | 0.4325 ± 0.0042 | 0.4342 ± 0.0044 | 0.4185 ± 0.0125 | none |
| CB (no-LoRA) | 0.4384 ± 0.0042 | 0.4365 ± 0.0044 | 0.4537 ± 0.0126 | 0.4388 ERA, same pipeline |

The eight subjects (college_biology, high_school_biology, virology, anatomy,
clinical_knowledge, medical_genetics, college_medicine, professional_medicine)
were chosen by subject name as topically adjacent to the WMDP-Bio forget domain,
following plan review point 8. They are general biology and medicine, not the
hazardous content WMDP-Bio targets, so they are adjacent to the forget domain
rather than part of it. The exclusion exists so the retain check is not depressed
by intended forgetting. It moved every model by at most 0.2 points. The
eight-subject column is reported separately as a check on spillover. 1565 items
removed, 12477 remain.

Aggregates computed by scripts/mmlu_aggregate.py, pooled binomial SE.

Checks: the script's full-set mean matches the harness `mmlu` group score to 1e-9
for all three models. The base full-set mean was recomputed independently in a
fresh Python shell from the per-subject `acc,none` and `sample_len` values
(0.4499, n=14042, match). The excluded count was verified by summing the eight
subjects' sample sizes (1565, match).

Interpretation: base is highest. CB and filtered sit 1.2 and 1.7 percentage
points below it (1.9 and 2.9 combined SE), and are within one SE of each other,
so their order is not established by the full set. Both interventions cost a
small amount of general capability. A drop of this size is consistent with either
a targeted intervention having some spillover or with ordinary run-to-run
variation between differently trained models, and this measurement cannot
separate the two. On the eight adjacent subjects, filtered sits 4.7 points below base (about 2.6 combined SE) while CB sits 1.2 points below (under one SE). The pretraining filter cost general biology knowledge that the fine-tune did not, which is consistent with the fine-tune being targeted at the hazard corpus rather than biology broadly.

Fail-fast 1, MMLU condition: passed. All three models are intact on general
capability. Degradation relative to base is under 2 percentage points for both
interventions.


## T4. Pre-caching verification: TransformerLens vs HF residual check

2026-08-28. Pod: A100 PCIe 80GB, US-KS-2. Script: scripts/check_tl_vs_hf.py.

Compared the final-layer residual stream from TransformerLens
(from_pretrained_no_processing) against a direct HF forward pass on
EleutherAI/deep-ignorance-unfiltered, same tokenization, single prompt built
exactly as the eval builds it (description line, question, correct candidate).

Architecture confirmed from config: 32 layers, hidden size 4096. Hook pairing:
TL blocks.31.hook_resid_post corresponds to HF hidden_states[32] (the entry
before the final layer norm).

| run | dtype | max abs diff | threshold | result |
|---|---|---|---|---|
| item 0 | float32 | 1.1e-4 | 1e-3 | pass |
| item 0 | bfloat16 | 3.9e-3 | 5e-2 | pass |
| item 5 | float32 | similar magnitude to item 0 | 1e-3 | pass |

The bf16 difference is consistent with bf16 rounding (about 3 significant
digits at residual scale), not a different computation. The item-5 run confirms
the comparison is real rather than vacuous (an exact zero would have indicated
comparing a tensor with itself).

Two environment notes for the caching script (for 2026-08-31 Monday):
1. TransformerLens 3.7.3 touches torch.backends.mps at import on CUDA-only
   boxes (upstream issue, fixed in a later TL release). The check script
   carries a runtime guard. The caching script needs the same guard until TL
   is upgraded.
2. TL's loader resolves hub metadata even with weights cached, so
   HF_HUB_OFFLINE=1 breaks it. Workaround: load via the local snapshot path,
   or set the offline flag only after models are loaded.


### T4b. Re-verification and caching on the RTX PRO 6000 pod (2026-08-31)

Pod: RTX PRO 6000 (Blackwell, 96GB), driver CUDA 13.2. Python 3.12.3,
torch 2.13.0+cu130, transformers 5.15.1, TransformerLens 3.7.3, resolved by
bootstrap from an unpinned requirements.txt. The 2026-08-28 check above ran on
an A100 PCIe under a manually installed cu126 torch build, a per-pod fix that
was not carried forward. The check was rerun so the verification and the
cached activations come from the same environment.

#### Block 31, previously unverified

The check compares TransformerLens against a direct HF forward pass. The two
run the same math in different implementations, so they cannot agree exactly:
floating-point addition is not associative, and summing the same 4096 terms in
a different order gives slightly different answers. The check asks whether the
gap stays at the size rounding would produce, or is large enough to mean the
two are computing different things.

Differences are reported in ULPs (units in the last place). A ULP is the gap
between two adjacent representable floating-point values at a given magnitude,
so a disagreement of 1 ULP means two numbers are as close as the format allows
without being equal. The gap is proportional to magnitude: about 0.4% of the
value in bfloat16, about 0.00001% in float32.

The 2026-08-28 run compared 31 of the 32 raw residuals. HF GPT-NeoX does not
put block 31's raw output in hidden_states, so it had no direct counterpart.
The script now registers a forward hook on the last layer to get it. All 32
raw residuals are verified.

A second change: the script can now run in float32 as well as bfloat16, which
makes it possible to test whether a disagreement is rounding. The weights are
stored in bfloat16, so running in float32 uses bit-identical weights and only
makes the arithmetic more precise. If a gap is rounding, it shrinks. If the
two implementations are doing something different, it does not.

Under the criterion in use at the time, block 31 failed, along with the row
comparing the output of the final LayerNorm.

| dtype | block 31 max abs diff | relative | post-final-LN max abs diff |
|---|---|---|---|
| bfloat16 | 16.0 | 2.48e-2 | 2.21 |
| float32 | 9.77e-4 | 1.52e-6 | 9.82e-5 |

Block 31's disagreement drops by more than four orders of magnitude, from 16.0
to 9.8e-4, with the same weights and more precise arithmetic. That is what
rounding does. A wrong weight or a different operation would stay about the
same in relative terms. Two supporting observations: in float32, blocks 22
through 30 all read exactly 4.8828e-4 and block 31 reads exactly 9.7656e-4,
one ULP larger, so the gap sits on the representable values rather than
growing with depth; and the float32 LayerNorm value (9.82e-5) reproduces the
2026-08-28 float32 result (1.1e-4).

The per-position diagnostics point the same way. In bfloat16 the largest
disagreement at any position is 16.0. Block 30's values reach about 3856, and
16.0 is about 1 ULP at that magnitude, so block 31's disagreement is the size
of what it inherited from block 30 rather than something block 31 introduced.
Position 0 is not the source (8.0, half the maximum); the maximum sits at
position 9.

#### The criterion

The 2026-08-28 run used flat thresholds (1e-3 float32, 5e-2 bfloat16). Those
were later restated as a relative criterion: a layer matched if the
disagreement was within 4 ULPs of the largest value in the tensor. Block 31
failed that at 6.4 ULPs. The criterion has been changed back to the flat
thresholds, with the ULP count kept as a diagnostic column.

Two reasons. First, the threshold was tighter than the comparison can deliver.
Error from summing 4096 terms in a different order accumulates to roughly 64
ULPs, so 4 was below the noise floor. Second, and more important, the
criterion measured the disagreement against the largest value anywhere in the
tensor, which is the token at position 0. That token carries a much larger
residual than the rest, about 3900 against 300 to 600 at the positions the
probe reads. Dividing by it made every layer look close. Block 31 shrinks the
position-0 residual to 644 and loses that large denominator, which is when the
tightness became visible.

The flat thresholds are not tighter or better calibrated. What justifies them
is that this check only needs to catch a computational difference before
caching, and the float32-versus-bfloat16 comparison shows it can: every layer
passes in both, and the gap shrinks with precision as rounding does.

Limitation: the flat thresholds divide by the same position-0 value, so they
are loose for the positions the probe actually reads.
scripts/check_batch_padding.py takes its maximum over the cached positions
only and avoids this. The check was left as is because the
float32-versus-bfloat16 comparison already establishes what it needs to
establish.

Environment sensitivity: the same script on the same prompt gave different
per-layer numbers on the two pods (layer 1: 1.09e-2 relative on 2026-08-28,
5.43e-3 today). Different torch builds use different kernels, which sum in
different orders. The recorded numbers are specific to the environment that
produced them.

Logs: results/eval/check_tl_vs_hf_20260831.log,
check_tl_vs_hf_20260831_diag.log, check_tl_vs_hf_20260831_diag_fp32.log.

#### Batching and right-padding

Each item is a question with four candidate answers, and per T2 each candidate
gets its own forward pass: the prompt is the question plus that candidate's
text, and the probe reads the state at the end of the candidate. So one item
produces four sequences that share a question and differ only in the appended
answer. For main-set item 0 those four are 43, 53, 51 and 47 tokens long.

cache_activations.py processes --batch-items questions per forward pass,
carrying each question's four candidates, so 32 sequences at the default of 8.
A batch has to be one rectangular tensor, so the shorter sequences are
right-padded with the eod of sequence id (NeoX has no pad token) up to the longest in the
batch. No attention mask is used.

Padding should not reach the cached values. Attention is causal, so a token at
index i attends only to indices at or below i, and right-padding puts the
filler after every real token, outside that window. Positions are rotary and
each token's phase depends only on its own index, which padding does not
shift. All three cached positions sit at or before the end of the real text.
The padded positions do get computed, but nothing is read from them.

Tested by caching the same 4 main-set items (16 sequences) twice and comparing
(scripts/check_batch_padding.py). Every batch pads, since an item's four
candidates differ in length, so the test varies the padding rather than
removing it. At --batch-items 1 each forward pass carries one question's four
candidates and pads to the longest of them. At 8 a pass carries all four
questions' 16 sequences and pads to the longest of those, so both the amount
of padding and which sequences share a batch differ between the two runs. If
padding were reaching the cached positions, that change should show up in the
activations.

- per-item maximum difference 4.0 for all four items, against per-item maxima
  of 536 to 576, with no concentration in the item whose candidates vary most
  in length
- the ratio of difference to residual size is flat across all 32 layers,
  3.4e-3 to 1.5e-2, with no entry point and no trend with depth
- per-position maximum difference 2.0 / 4.0 / 4.0 for q_end / answer_marker /
  cand_end
- worst ratio 1.493e-02, about 3.8 bfloat16 ULPs of that layer's largest value

The denominator here is the maximum over the cached positions only, so unlike
the TL-vs-HF criterion it is not inflated by position 0.

The differences are at rounding scale, at most 3.8 bfloat16 ULPs of the layer's largest cached-position value. More importantly they show no structure. A padding effect would enter at a specific layer and grow with depth as a fraction of the residual. The ratio instead varies without trend between 3.4e-3 and 1.5e-2 across all 32 layers, the four items show the same difference at bfloat16 resolution regardless of how much padding each receives, and no cached position stands out. Different batch shapes change kernel choice and summation order, which accounts for differences of this size.

Two limitations. A padding effect smaller than one bfloat16 ULP at the cached
positions would be indistinguishable from rounding here. And the test varies
the padding pattern rather than removing padding, so an effect identical
across both patterns would not show.

The cache is therefore not bit-identical across --batch-items settings. The
full run used the default, 8.

Log: results/eval/pad_batch_check_20260831.log.

#### Caching run

Three checkpoints and two item sets of 1076 items each, with four candidates
per item, so 3 x 2 x 1076 = 6456 item-model-set combinations and 25,824 forward passes. 
Each set was written in chunks of 128 items, so nine files per model per set:
eight full chunks and a final chunk of 52. 54 files in total, 18.92 GiB on the
volume. Forward time 2:39, measured around the batched forward pass including the
per-layer copy of the selected positions back to CPU. Model loading,
tokenization, and writing to the volume are outside that figure.
The revision hashes in each meta.json match T1: c8df368f (unfiltered), b28797cd (e2e-strong-filter), c57ec059
(unlearned-cb). Each chunk carries its revision, dtype, torch, transformers
and TransformerLens versions, item set file and seed, position indices and
names, and the prompt template.

Positions were located from the tokenized prompt, not assumed. The script asserts for every sequence that cand_end is the last token, and prints the tokens at all three positions for the first three items. Within an item, q_end and answer_marker are the same across the four candidates while cand_end moves with candidate length, as T2 assumes. Each chunk is checked for NaNs before writing.

Per-position norm tables for all six model-set combinations are in
results/eval/cache_full_20260831.log. Figures parsed from that log by
scripts/plot_norms.py are in results/: one per model-set combination, plus
cross-model comparisons on pos0 and cand_end for each set.

This check is at a high level: whether the activations read through
TransformerLens are reasonable (no NaNs, sane magnitudes, positions landing on
the right tokens). The numbers are means over the first batch of 32 sequences
and are aggregated within a layer, so they carry little information on their
own. What they show is the general shape: norms grow with depth at the cached
positions, and position 0 is orders of magnitude larger than the rest.

Observations, main set:

- Position 0 rises steeply through layers 4 to 8 and is flat from about layer
  10 onward, at roughly 4400 for base and CB and 5900 for the filtered model.
  Through that plateau it is ten to twenty times the cached positions, which
  run 40 to 400. This is the basis for the T2 decision to exclude position 0
  from anything averaged over positions.
- At layer 31 position 0 collapses (4220 to 218 for base, 5512 to 230 for
  filtered, 4220 to 221 for CB) while all three cached positions rise. For
  base, q_end goes 323 to 499, answer_marker 355 to 659, and cand_end 372 to
  606.
- All three models jump at layer 6, but the filtered model jumps further:
  452 to 4554, most of the way to its eventual plateau, while base and CB go
  937 to 2547 and then climb for four more layers. Filtered also sits below
  the other two at layers 4 and 5 before overshooting them.
- Base and CB match to within a few tenths at position 0 at every layer. They
  differ at cand_end, where CB runs higher from about layer 17 (401.6 versus
  372.4 at layer 30, 694.7 versus 606.0 at layer 31). With no uncertainty
  attached to a single-batch mean, this is an observation rather than an
  established difference.  

What these do and do not support. The filtered model was pretrained
separately, so it does not share a weight coordinate frame with base and its
internal scales are not expected to match. A norm difference between filtered
and the other two is close to uninterpretable on its own. Base and CB do share
a frame, since CB was fine-tuned from base, so the cand_end difference between
them is at least a valid comparison, though a norm difference alone says
nothing about what is encoded.

These norms do not carry into the probe results. The probe standardizes each
of the 4096 coordinates using per-model means and standard deviations, so
overall scale differences between models are removed before fitting. Where the
position-0 magnitude did matter was in the TL-vs-HF check above, whose
criterion divided by the largest value in the tensor.

Log: results/eval/cache_full_20260831.log.

#### Extraction for T5

The cand_end position was extracted into 32 per-layer files, one per layer,
each holding all three models and both item sets, 201.8 MiB each and 6.31 GiB
total, at /workspace/cand_end_layers. scripts/check_extraction.py compared all
32 layers against the source chunks: bit-identical for all six model and set
combinations, with item indices and revision hashes matching.

Logs: results/eval/extract_cand_end_20260831.log,
check_extraction_20260831.log.

Note for T5: activations are cached in bfloat16, so block 31 carries about 1%
noise relative to the residual size at the positions the probe reads. The
probe standardizes features per model, so this is unlikely to affect the
comparison, but it belongs in limitations.


## T5 working notes (2026-09-01, superseded by the T5 section below)

### Sanity Notes
Shuffled-label baselines over all 32 layers and three models. Within-item
shuffle (seed 1001), read on argmax4 against an analytic chance of 0.25: main
mean 0.2473 (range 0.195 to 0.316), control mean 0.2451 (0.189 to 0.338).
Global shuffle (seed 1002), read on AUC against 0.5: main mean 0.5127 (0.491
to 0.539), control mean 0.5006 (0.478 to 0.533).

Argmax4 baselines sit at chance. The main-set AUC baseline runs about 1.3
points above 0.5 on average rather than scattering evenly around it, so small
AUC differences on the main set should be read against that offset rather than
against 0.5.

Regularization strength chosen by CV: on the main set, 92 of 96 fits selected
1e-3 or 1e-4. On the control set the selection is spread, with 13 fits at C of
1 or above. The lbfgs convergence warnings are concentrated in those weakly
regularized fits, which are also the ones where CV found little to prefer.


### Baseline for the plots

The shuffled baselines use a fixed permutation, drawn from seed 1001 and reseeded identically for every layer and model, so the same scramble is applied throughout. This makes them reproducible but means the dashed curves are one draw rather than an average: their layer-to-layer wiggle is the same permutation being scored at different depths, not independent noise, and the three models' baselines are correlated with each other. The scatter is consistent with the binomial standard error of about 0.024 on 323 held-out items.

## Observation interpretation

Why main is higher: WMDP-Bio Verified Cloze items were built with verified distractors for a knowledge benchmark, while the control items are MMLU questions reformatted into a cloze shape. So the two sets differ in construction, not just in content, and a level difference between them isn't interpretable on its own. What the control is for is the model-to-model comparison within each set, not the absolute level.

Why the control's transition is sharper: I don't have an account for that. Worth recording as an observation. One thing that might bear on it: the control's AUC is near chance until layer 10 and rises steeply through 12 to 18, while main is already at 0.58 at layer 0. So main has signal available at the embedding, which fits items whose distractors were selected to be discriminable, while control needs more depth to build it.



# T6 candidates, noted 2026-09-01:
- Could the probe be reading properties of the candidate text alone rather
  than question-candidate fit? Correct and wrong candidates come from the
  same pool of biological terms and each item's correct answer ends in a
  different token, so there is no obvious token-level shortcut, but this is
  testable: fit a probe on the candidate text without the question and see
  whether it separates.


## T5. Probe curves and baselines

2026-09-01. Pod: L40 48GB. Scripts: probe_one_layer.py (pre-check),
probe_sweep.py (full run), probe_bootstrap.py (uncertainty),
plot_probe_sweep.py (figures).

### Setup

Per T2: linear probe on the residual stream at cand_end, one probe per
(layer, item set, model). Each item contributes four examples, one per
candidate, labelled by whether that candidate is correct. Item-level
70/30 split from the T3 id lists, seed 42 (main) and 43 (control), so an
item's four candidates never straddle the split. Features standardized
using means and standard deviations from that model's training items
only. Regularization strength chosen by 5-fold GroupKFold cross-
validation grouped by item, inside the training items, over a C grid of
1e-4 to 1e2.

Metric changed from accuracy to within-item argmax with AUC alongside,
and CV scoring changed from accuracy to AUC. See plan.md Deviations
2026-09-01 and T2 Metric. The reason follows from the label structure
rather than from the results: one correct candidate per three wrong by
construction means a constant "not correct" prediction scores 0.75, and
accuracy-scored CV selected the strongest regularization in the grid for
all three models, collapsing the probe to that constant prediction.

Coverage: 32 layers x 2 item sets x 3 models = 192 cells.

### Headline numbers

Figures in results/figures/probe/: argmax4 and AUC for each item set,
with model curves, per-model shuffled baselines, the analytic chance
line, and the T2 intervention layers marked.

Main set. All three models start above chance at layer 0, about 0.35 to
0.37 argmax4 against 0.25, and rise to a plateau from roughly layer 15
onward. AUC starts near 0.58 and plateaus around 0.62 to 0.70.

Control set. Lower throughout and with a sharper transition. AUC sits at
chance until about layer 10, then rises through layers 12 to 18 to a
plateau. Argmax4 runs about 0.26 to 0.31 early and 0.37 to 0.44 in the
plateau, from layer 18 onward.

Values from probe_sweep.json. The range columns are over layers 15 to 31,
the same window as the gap summary below. On the control set that window
opens inside the transition, so its low end is a layer-15 value rather
than a plateau value.

| set | model | L0 argmax4 | L20 argmax4 | L31 argmax4 | L15-31 argmax4 | L20 AUC | L15-31 AUC |
|---|---|---|---|---|---|---|---|
| main | base | 0.365 | 0.517 | 0.508 | 0.483-0.533 | 0.700 | 0.670-0.700 |
| main | filtered | 0.347 | 0.458 | 0.449 | 0.421-0.480 | 0.673 | 0.623-0.673 |
| main | fine-tune | 0.362 | 0.489 | 0.486 | 0.471-0.523 | 0.683 | 0.666-0.690 |
| control | base | 0.307 | 0.430 | 0.421 | 0.365-0.440 | 0.629 | 0.582-0.634 |
| control | filtered | 0.276 | 0.406 | 0.399 | 0.328-0.415 | 0.610 | 0.554-0.610 |
| control | fine-tune | 0.263 | 0.437 | 0.399 | 0.356-0.440 | 0.621 | 0.579-0.625 |

The model ordering is base above fine-tune above filtered on the main
set. Whether any of those separations is resolved is a question the
curves cannot answer, since these are point estimates on a shared
held-out set; see Model gaps and the paired bootstrap below. Mean gaps
over layers 15 to 31 are +0.048 argmax4 and +0.037 AUC for base minus
filtered on main, and +0.012 argmax4 and +0.006 AUC for base minus
fine-tune.

### Uncertainty

Two axes per T2, from probe_bootstrap.py.

Held-out items: 1000 resamples at the item level so an item's four
candidates move together, seed 2001, all 32 layers and both sets. 2.5
and 97.5 percentiles reported. Half-width about 0.024 on argmax4.

Probe-training subsamples: 30 resamples of the training items with
replacement, refit at the C already chosen by the sweep, seed 2002, all
32 layers, main set. Reported as a standard deviation rather than a
percentile interval, since 30 draws cannot support a stable tail
percentile. Argmax4 standard deviation 0.015 to 0.024, AUC 0.006 to
0.010, flat across layers and models.

The two axes are comparable in size. The plotted bands are the item axis
only and therefore understate total uncertainty; the figures carry a
caption saying so.

Neither axis is training-run variance. Both hold the trained models
fixed. See Limitations.

Reusing the swept C rather than redoing CV inside each resample excludes
C-selection variance from the reported spread. On the main set 92 of 96
fits selected 1e-3 or 1e-4, so the omitted variance is likely small
there; on the control set the selection is scattered and the assumption
is weaker.

### Sanity checks

Shuffled-label baselines, over all 32 layers and three models. Within-
item shuffle, seed 1001, read on argmax4 against an analytic chance of
0.25: main mean 0.2473 (range 0.195 to 0.316), control mean 0.2451
(0.189 to 0.338). Global shuffle, seed 1002, read on AUC against 0.5:
main mean 0.5127 (0.491 to 0.539), control mean 0.5006 (0.478 to 0.533).

Argmax4 baselines sit at chance. The main-set AUC baseline runs about
1.3 points above 0.5 on average rather than scattering evenly around it,
so small AUC differences on the main set should be read against that
offset. The baselines use a fixed permutation reseeded identically at
every layer and model, so the dashed curves are one draw rather than an
average: their layer-to-layer variation is the same scramble scored at
different depths, not independent noise, and the three models' baselines
are correlated.

Regularization selection. Main set: 92 of 96 fits chose 1e-3 or 1e-4.
Control set: scattered, with 13 fits at C of 1 or above. The lbfgs
convergence warnings are concentrated in those weakly regularized
control fits.

Fixed-C refit cross-check. probe_bootstrap.py refits at the C recorded
by the sweep and warns if the resulting point estimate differs from the
sweep's by more than 0.005. Ten warnings, all on the control set, none
on main. The divergences are in the weakly regularized cells where lbfgs
does not converge, so the same nominal fit can land in different places.

Logs: results/eval/probe_one_layer_20260901.log,
probe_sweep_20260901.log, probe_bootstrap_20260901*.log.
Data: results/probe_sweep.json, probe_bootstrap.json,
probe_heldout_scores.npz.

### Model gaps and the paired bootstrap

2026-09-02. Script: gap_tables.py. Log:
results/eval/gap_tables_20260902.log. Data:
results/figures/probe/gap_paired_bootstrap.json. Figures: gap_auc.png,
gap_argmax4.png.

The per-model bands above are conservative for a model-to-model
difference, because the item-axis draws are identical across the three
models of a cell, so the shared item variation cancels in a paired
comparison. gap_tables.py recomputes the item bootstrap from the saved
held-out scores, taking the base-minus-model difference per draw and
reporting the 2.5 and 97.5 percentiles of those differences. The draws
reproduce probe_bootstrap.py's: default_rng([2001, layer, iset]), 1000
resamples. As a check, the per-model percentiles recomputed on the way
match those stored in probe_bootstrap.json exactly.

Summary over layers 15 to 31, 17 layers. "resolved" means the paired
2.5th percentile is above zero.

| comparison | metric | positive | resolved | mean gap |
|---|---|---|---|---|
| base - filtered, main | AUC | 17/17 | 12/17 | +0.037 |
| base - filtered, control | AUC | 17/17 | 13/17 | +0.026 |
| base - fine-tune, main | AUC | 12/17 | 7/17 | +0.006 |
| base - fine-tune, control | AUC | 17/17 | 9/17 | +0.011 |
| base - filtered, main | argmax4 | 17/17 | 7/17 | +0.048 |
| base - filtered, control | argmax4 | 16/17 | 1/17 | +0.028 |
| base - fine-tune, main | argmax4 | 11/17 | 2/17 | +0.012 |
| base - fine-tune, control | argmax4 | 12/17 | 1/17 | +0.010 |

AUC resolves what argmax4 does not, as expected: argmax4 collapses the
1292 held-out examples into 323 item-level decisions, so its intervals
run about 1.7 times wider on the same data.

Last layers, base minus fine-tune on main AUC: the gap is negative from
layer 26 onward at five of the last six layers (layer 27 is +0.001),
with magnitudes 0.006 to 0.010, so the fine-tune reads slightly above
base. Every one of those intervals includes zero. The control gap stays
positive across the same layers. Recorded as an observation, not a
result.

### Reading

1. The probe reads well above what the models output, on every model.
   At layer 20 on main, argmax4 is 0.517 base, 0.458 filtered, 0.489
   fine-tune, against Cloze scores of 0.3652, 0.2435 and 0.2537 in T1.
   That gap between internal readability and behavior is what the
   project was built to look for.

2. But the removal reference does not read low enough, so there is no
   floor. Filtered reads 0.42 to 0.48 on main-set argmax4 while scoring
   0.2435 behaviorally. The same high reading appears on the model built
   to lack the material, so a high probe score cannot indicate retained
   knowledge, and the readability-versus-behavior gap in point 1 is not
   evidence of suppression. Without a floor there is too little span to
   place a third model within.

   Note on what filtered is. No Deep Ignorance model was trained on the
   forget corpus; those papers were used to derive the blocklist
   keywords. Filtered was pretrained on a corpus with keyword-matched
   documents removed, which is a weaker condition than never having seen
   the material.

3. Base minus filtered is resolved from about layer 15, and the same gap
   appears on the control set. On AUC the paired interval excludes zero
   at 12 of 17 layers on main and 13 of 17 on control, with mean gaps of
   0.037 and 0.026. On argmax4 the direction is the same, positive at 17
   of 17 on main and 16 of 17 on control, but resolved at only 7 and 1.
   T2 committed in advance that similar patterns on both sets would
   undermine the knowledge reading rather than support it. Similar
   patterns on both is what obtained.

4. The most likely explanation of that resolved gap is general biology
   the filter removed alongside its target, rather than the targeted
   knowledge. Two observations point there: the control gap above, and
   the T1 result where filtered scores 0.047 below base on the eight
   MMLU bio/med subjects while the fine-tune scores 0.012 below. Those
   are the same effect through a probe and through a benchmark on
   overlapping subjects, so they are not independent evidence. The
   mechanism is what makes it plausible: the blocklist ran at 44 to 58%
   precision, was applied without classifier review, and was never
   measured on the training corpus itself.

5. Base minus fine-tune is not resolved. Mean AUC gaps from layer 15 are
   0.006 on main and 0.011 on control, against 0.037 and 0.026 for base
   minus filtered. Three explanations remain open and this design cannot
   separate them:
   - The fine-tune changed little that the probe reads. The intervention
     was one epoch on 1,024 documents, about 28 optimizer steps.
   - The instrument lacks the resolution. The gap may be real and
     smaller than this design measures.
   - The instrument is structurally blind to this intervention. The
     circuit-breaker objective pushes the fine-tune's representations of
     forget text to be orthogonal to base's representations of the same
     text, and a probe fit separately per model is invariant to
     rotations of that model's own representation space.

6. Which gate this triggered. Not fail-fast 3, which required that no
   layer separate base from filtered above the bootstrap spread: 12 of
   17 layers do on AUC. What obtained is fail-fast 2, the control test.
   The target gives a similar result on the control set as on the main
   set, so it cannot be attributed to the targeted knowledge rather than
   to biology in general.


## T7. Sanity checks

Grouped by what could have gone wrong rather than as a flat list.

### Did the intended checkpoints load

Weight revision hashes recorded in every chunk's meta.json and matching
T1: c8df368f (unfiltered), b28797cd (e2e-strong-filter), c57ec059
(unlearned-cb). Architecture read from the config rather than assumed
(32 layers, hidden size 4096), and the TransformerLens config asserted
against the HF config on n_layers, d_model and d_vocab before any
caching. See T1 and T4b.

### Does TransformerLens compute what the model computes

All 32 raw residuals compared against a direct HF forward pass, with
block 31 reached via a forward hook since GPT-NeoX does not expose it in
hidden_states. The disagreement was tested for whether it is rounding by
rerunning in float32 on bit-identical weights: the absolute error fell
from 16.0 to 9.8e-4. See T4b, including the criterion that was reverted
and why.

### Do the reported numbers correspond to the data

Positions located from the tokenized prompt, with an assertion for every
sequence that cand_end is the last token, and the tokens printed for the
first three items. NaN check on every chunk before writing. The
per-layer extraction verified bit-identical against the source chunks
for all 32 layers and all six model-set combinations.

Independent recompute: argmax4 at layer [N] for [model] computed
directly from the saved held-out scores in probe_heldout_scores.npz,
without passing through the metric code used by the sweep.
  sweep value: [FILL]
  recomputed:  [FILL]
Script: scripts/show_examples.py.

### Could the signal be an artifact of the procedure

Shuffled-label baselines at their analytic chance levels across all 32
layers and three models (T5). Item-level splitting so an item's four
candidates never straddle the split, and item-grouped CV folds for the
same reason. Standardization from training items only. Regularization
chosen inside the training items. Control set run as a topic-matched
comparison, with the caveat in Limitations that it is affected by the
same filter.

### Reading the data

Ten held-out items read by eye, selected with seed [N] from
main_set.json. The items are coherent four-way questions with plausible
distractors, several differing by a single term (Cor a 2 vs Cor a 8,
Agrobacterium vs E. coli, synthesizes vs cleaves). Correctness was not
independently verified: WMDP-Bio was expert-constructed and the reader
is not a domain expert.

Observations worth recording. One item is a history question with no
biology content (which facility became the primary location for the U.S.
Army Biological Warfare Laboratories in 1943). One presents four
near-identical 150-word paragraphs differing in one or two terms. The
paper reports that selecting the longest answer scores 46% on the full
WMDP-Bio set, and the Verified Cloze subset was curated in part to
reduce shortcut exploitation; no length check was run here.

Two items included in the write-up. See writeup.md.

### Ten items with probe outputs

[FILL from show_examples.py]   