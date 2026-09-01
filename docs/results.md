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
per item, so 3 x 2 x 1076 = 6456 item-model-set combinations passes and 25,824 forward passes. 
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

Per-position norm check, unfiltered / main (mean L2, first batch):

layer       pos0         q_end answer_marker      cand_end
    0       48.9          35.2          34.1          39.2
    1       64.4          43.0          40.0          50.2
    2      156.7          43.2          40.3          55.3
    3      190.9          41.7          38.6          58.0
    4      408.4          39.8          37.7          60.0
    5      936.8          40.2          38.9          62.5
    6     2546.9          46.3          43.4          67.2
    7     3472.8          51.2          45.3          71.3
    8     3787.1          54.7          49.4          77.2
    9     3974.6          57.4          52.5          81.2
   10     4190.2          63.9          52.3          86.6
   11     4256.5          69.6          53.5          93.8
   12     4298.3          76.9          57.9         102.1
   13     4323.2          84.7          65.5         110.0
   14     4345.6          88.1          72.4         118.7
   15     4368.2          99.3          79.5         128.4
   16     4387.8         105.7          85.0         131.1
   17     4393.2         111.4          92.7         136.1
   18     4393.4         113.9          96.2         141.6
   19     4393.4         124.1         103.3         144.4
   20     4393.3         131.8         110.3         150.4
   21     4393.6         139.3         119.8         154.2
   22     4393.7         149.8         133.5         161.1
   23     4393.7         163.4         147.7         169.1
   24     4393.5         174.2         164.0         181.5
   25     4396.8         194.4         187.0         205.2
   26     4400.1         224.5         221.8         229.4
   27     4383.1         246.9         250.3         252.4
   28     4333.0         279.1         289.6         293.4
   29     4300.4         294.6         314.0         325.8
   30     4219.8         323.2         354.5         372.4
   31      218.1         498.8         658.6         606.0
[unfiltered/main] items 8/1076  23.13 items/s  est. remaining (all selected runs, excl. model loads) 0:04:38  cumulative written 0.0 MiB

Plots for all six model-set combinations, plus cross-model comparisons on
pos0 and cand_end, are in results/ (scripts/plot_norms.py, parsed from the
caching log).

DRAFT READING, NOT YET REVIEWED [check tomorrow]:

These norms are a diagnostic, not a result. A norm collapses a 4096-dimensional
vector to one number, so it describes scale rather than content, and per T2 the
probe reads direction after standardizing features. Two models could have the
same norm curve and encode different information, or differ in norm while
encoding the same thing.

Position 0 rises steeply through the early layers, plateaus around 4400 from
about layer 10, and collapses to 218 at layer 31. It runs roughly ten to a
hundred times larger than the three cached positions throughout the plateau.
This is the attention-sink pattern, and it is why T2 excludes position 0 from
anything averaged over positions.

The three cached positions grow roughly monotonically with depth, from about
35 at layer 0 to 500-660 at layer 31, with cand_end above q_end above
answer_marker for most of the network. All three rise at layer 31, the same
block where position 0 collapses.

Across models, on the first-batch sample:
- base and CB are nearly indistinguishable on position 0, which is expected
  since CB was fine-tuned from base.
- e2e-strong-filter's position 0 is distinctly different: it jumps at layer 6
  to about 4550 and plateaus around 5900, against a gradual climb to about
  4400 for the other two.
- on cand_end, CB sits slightly above base and filtered from about layer 17
  onward, with the gap widening toward layer 31 (694, 606, 597).
- filtered's cand_end is slightly elevated over layers 1 to 5, converging by
  layer 7.

Caveat: these are means over the first batch of 32 sequences, not the full
1076 items, so the small cross-model differences at the cached positions have
no uncertainty attached and should not be read as established. The position-0
difference for filtered is large enough to be visible regardless.

TODO [YOU]: the reading. Material: position 0 climbs to about 4400 and
flattens from layer 16, which supports the T2 decision to keep it out of
anything averaged over positions. At block 31 position 0 drops to 218 while
the three cached positions rise (q_end 323 to 499, answer_marker 355 to 659,
cand_end 372 to 606). Across models: e2e-strong-filter's position 0 runs about
5900 from layer 16 against about 4400 for base and CB, and rises sharply at
layer 6 rather than climbing gradually. Base and CB are close to identical,
which is expected since CB was fine-tuned from base while the filtered model
was pretrained separately.

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