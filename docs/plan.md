# Execution Plan

## Purpose
Divide the MATS task into tickets with time estimates so progress is trackable and
the budget is visible. Per Neel's doc: write the plan with estimates, then don't
take it seriously. Estimates use the measured day shape (about 3h of blocks per
day) and a 1.5x factor on tickets involving new code, from the first two days'
estimate misses.

This plan was reviewed once in a fresh session with an adversarial prompt. The
review is in docs/plan_review_20260826.md. Changes made in response are marked
[rev] below. Items not addressed are in the write-up's limitations.

## Budget
- Neel's cap: 20h counted + 2h for the executive summary
- Counted so far: 1:35 (planning, eval spec)
- Remaining under cap: ~18.4h
- Availability through Sep 3: ~16.5h (the binding constraint)
- Tickets T1-T7 + T9 + buffer sum to 16.5h. T8 is out by default.
- Target: experiments done and first probe curve by Fri Aug 28. Submission Sep 2-3.
  Sep 4 is buffer, not the plan.

## Tooling note
TransformerLens for activation access. Neel's doc suggests nnsight or raw PyTorch
hooks. TL is used here because the smoke test already runs on it, the three
checkpoints load, and the measurement is comparative across checkpoints through
one tool, so consistency matters more than exact fidelity to the HF forward pass. [rev] All three models are loaded with `from_pretrained_no_processing`. The default `from_pretrained` applies `center_writing_weights` and `fold_ln`, which are per-checkpoint transforms on the residual stream being compared. Verified in T4 by checking one prompt's hidden state against an HF forward pass before caching. Fallback if TL hits a memory or conversion problem on the L40: raw forward hooks on the HF model, about 15 lines, not a migration.

---

## T0. Planning: 2h 30m
- Goal ladder and fail-fast (0:50, Aug-24), eval_spec.md (0:45, Aug-25), plan.md with review (0:55, Aug-26)

## T1. Behavioral anchor: 2h
- Description: Cloze (wmdp_bio_cloze_verified) and MMLU on base, e2e-strong-filter,
  and no-LoRA CB through lm-eval 0.4.12, using the vendored YAMLs. See
  docs/eval_spec.md for configs.
- Expectation: base high on both. Filtered and CB low Cloze, MMLU near base.
  CB's Cloze number is a replication of its selection metric, said once.
- Core framing: on these benchmarks CB looks like filtered. The question is
  whether internals say otherwise.
- MMLU: fixed random subset (a few hundred items) with logged seed, not the full
  14042. Note that --limit takes the first N per subject, so a subset file or an
  explicit caveat is needed.
- [rev] MMLU subset excludes bio-adjacent subjects (college_biology, virology,
  and medical subjects) so the retain check does not overlap the forget domain.
  Reported as the aggregate with its binomial interval, not as "near base."
- Deliverable: 3x2 table with checkpoint commit hashes and harness version, in
  the results doc.
- Gate: fail-fast 1. If base and CB are not clearly apart on Cloze, or MMLU is
  broken for any model, stop and debug before caching activations. Checks in
  order: weight hashes, tokenizer and prompt template, subset size, filter
  variant.
- Depends on: YAMLs vendored into lm_eval_tasks/ (uncounted, 10 min).

## T2. Decide what the probe measures and where: 1h [rev: was 0.5h]

Description: written decisions only, no code. Everything here is fixed before any
activations are saved, so that no choice is made after seeing the results.

### What a probe is here
A linear classifier (logistic regression) trained on the model's internal state at
one token position and one layer. Its input is the residual stream vector at that
position and layer, which is d_model numbers, standardized feature by feature using
means and standard deviations computed on the training items of that model. If the
classifier predicts above chance on held-out items, that information is present at
that position and readable by a simple linear readout.

Standardization is not the same as the L2 norm. The norm collapses the vector to
one number and would throw away the information the probe needs.

### What the probe predicts
Each evaluation item has four candidate answer texts, one correct and three wrong
(the wrong ones are called distractors). The probe's label is whether a given
candidate is the correct one.

### Decisions

**Label formulation.** Four forward passes per item, one per candidate answer.
The probe reads the state at the end of the candidate text and predicts whether
that candidate is correct. This gives 4304 examples per model per layer, one
positive and three negative per item. Chosen because it matches how the benchmark
itself scores and needs no extra machinery.

Alternatives considered and rejected: a single pass on the question alone with a
four-way label, which does not work because the label depends on a candidate
ordering the model never saw; and a single pass with a regression onto an
embedding of the correct answer, which reads further upstream but makes the
embedding model part of the instrument.

**Positions recorded.** All three of the following, from the same forward pass, so
the extra cost is storage rather than compute:
1. End of the question text.
2. The "Answer:" marker.
3. End of the candidate answer text.

Only position 3 carries a valid label under this formulation, since positions 1
and 2 are identical across an item's four passes. Position 3 is what the probe
uses. Positions 1 and 2 are recorded in case a different target is worth trying
later.

**Where the intervention acted.** Circuit breakers were trained with a loss on the
hidden states at layers 5, 10, 15, 20, 25, 30 (ERA config, matching Cas's
default). Those six indices are marked on every figure, and the analysis
pre-commits to looking for structure at them.

An important caveat: this was full fine-tuning, not LoRA. Every weight was
trainable, so the optimizer changed early-layer weights if that reduced the loss
measured at layers 5 and above. Early layers are therefore changed indirectly,
not untouched, and the layer axis gives a gradient of directness rather than a
clean split between touched and untouched.

(LoRA differs by freezing the original weights and adding trainable adapters at
chosen modules, so modules without an adapter stay exactly unchanged. Gradients
still flow through the whole network in both cases. The released Deep Ignorance
CB variant uses LoRA on the same base model, which makes it the natural
one-variable follow-up if the layer-axis result turns out to matter.)

**Controls.** Both of the following:
1. Shuffled labels on the same vectors, to establish the chance level.
2. A topic-matched item set, built from MMLU biology and medicine questions reformatted into the same prompt shape, to test whether the probe reads specific knowledge or general topic. This adds one caching pass per model.
  - What it tests: whether the probe reads specific knowledge or general biology topic.
  - Why these items: same domain, but not the material any intervention targeted.
  - How to read it: similar curves across the three models on the control while they differ on the main set supports the knowledge reading. Similar patterns on both undermines it.
  - Caveat: filtered scored several percentage points below base on those eight subjects, so the control is not perfectly neutral.
  

**Split.** 70/30 train and held-out, split at the item level so all four candidates of an item land on the same side. Without this the probe could see three candidates of an item in training and the fourth in held-out, which leaks. Class balance is automatic, since each item contributes one positive and three negatives. About 320 items held out, about 1290 examples. Seed 42.

**Layers.** All layers, reading the residual stream after each block. The layer
count is confirmed from the model config in T4 rather than assumed. Recording all
layers is affordable and removes the need to justify a subset.

**Items.** All 1076 cloze-compatible items, not only the ones base answers
correctly. Selecting on base behavior would bias the comparison.

**Position 0.** Excluded from any quantity that averages over token positions,
decided now on the basis of the smoke test finding that position 0 carries very
large activation norms and dominates such averages. This applies to the norm
diagnostic plots in T4. It does not affect the probe, which reads one specific
position rather than an average.

**Regularization.** The probe's regularization strength is chosen by
cross-validation inside the training items only, never using held-out items. Using
held-out data to pick a setting would leak information and inflate the score.

### How precise this can be
With about 320 held-out items, the standard error on a probe accuracy is roughly
3 percentage points, so a 95% interval is about plus or minus 6 points.
Differences between model curves smaller than that cannot be resolved by this
design and will not be claimed.

### What gets plotted
One figure per recorded position that has a usable label. X axis is layer index,
Y axis is probe accuracy on held-out items. Three lines, one per model, each with
an error band from resampling. A horizontal line at the shuffled-label level.
Vertical marks at layers 5, 10, 15, 20, 25, 30.

### What each possible outcome would mean
Written in advance so that no result is interpreted after the fact. 

Note before reading these: CB sitting near base is close to guaranteed by
construction, since CB was fine-tuned from base while filtered never saw the
data at all. A flat near-base curve is therefore a weak result. What the
experiment can actually discover is the shape: where along the layer axis CB
stops looking like base, and whether filtered shows the same shape. That is why
outcome (c) is the informative one.

(a) CB's curve sits on top of base at every layer.
    Reading: the fine-tune barely changed what is linearly readable. Information
    about the answer is still present. This says little about whether the model
    can actually use it.

(b) CB's curve sits on top of filtered at every layer.
    Reading: the information is not linearly readable at the layers measured.
    Consistent with the knowledge being gone, and also consistent with it being
    held in a form a linear probe cannot see.

(c) CB's curve depends on layer, for example tracking base at early layers and
    falling away at or after the intervention layers.
    Reading: the pattern most consistent with the knowledge being present but
    routed away from the output. The most informative outcome, and the one that
    would justify running a recovery experiment next.

### Deliverable
This section, filled in. Moves to docs/experiment_spec.md if it outgrows the plan.

### Stop condition
Fail-fast 2: if the chosen target would plausibly give the same result on all
three models because it measures topic rather than knowledge, redefine it before
proceeding. The topic-matched control set is what tests this.

### Note for T3
The control set adds work not in the original T3 ticket: MMLU biology and medicine
items need reformatting into the cloze prompt shape with four choices each. About
20 minutes.


## T3. Prompt sets: 1h
- Description: build the item sets and the split fixed in T2.
  1. Main set: all 1076 cloze_compatible items from EleutherAI/wmdp_bio_cloze. Split 70/30 train and held-out, split at the item level so all four candidates of an item land on the same side. Seed 42, logged in the output file.
  2. Control set: MMLU biology and medicine questions from the eight subjects listed in T1, reformatted into the same prompt shape with four choices each. Random 1076 of the 1565 available, drawn without replacement from a pool concatenated in the listed subject order. Sampled from the pool with seed 42 and split with seed 43, so the two partitions are visibly independent, same 70/30 split at the item level.
- Deliverable: item-id lists (753 train and 323 held-out for both sets), committed to the repo, plus the script that produced them.
- Gate: none. Sanity checks: item counts, no overlap between train and held-out, control items have four choices each and the same prompt format as the main set, one main item and one control item inspected side by side to confirm the same structure.

## T4. Activation caching: 3h [rev: was 2h, 1.5x applied]
- Description: for each of the three checkpoints and for both item sets (main
  and control), run four forward passes per item (one per candidate answer) and
  save the residual stream after every layer at the three positions fixed in T2.
  8608 passes per model. One model at a time, saved to the volume.
- Start with base: confirm the layer count and hidden size from the config,
  verify TransformerLens hook names and tensor shapes on the NeoX architecture.
- [rev] Before caching: load with `from_pretrained_no_processing`, run one fixed
  prompt through TransformerLens and through the HF model directly, confirm the
  final-layer residual matches to numerical tolerance. Log the flags used.
- Locating positions: the end-of-question token, the "Answer:" marker, and the
  end-of-candidate token found from the tokenized prompt, not assumed. Check on
  a few items by printing the tokens at those positions.
- Per-position norm check per model, with position 0 shown separately.
- Deliverable: cached tensors on the volume for three models and two item sets,
  the script, a per-layer norm plot per model, the HF-vs-TL check output.
- Gate: none formal. Sanity: shapes match expectation, positions land on the
  right tokens, norms look sane, no NaNs.


## T5. Probe curves and baselines: 4.5h [rev: was 3h, 1.5x applied]
- Description: per layer, per site, per model: standardize features (mean/std
  from that model's training split), fit logistic regression to the T2 target,
  score on held-out. Plot accuracy vs layer, three curves per figure, one figure
  per site.
- Baselines: shuffled-label probe (chance), control-set probe (should not
  separate if the target is knowledge rather than topic).
- [rev] Error bars from two bootstrap axes: over held-out prompts (item
  variance) and over probe-training subsamples (fit variance). Neither is
  training-run variance. Stated in limitations.
- Deliverable: the headline figures and the numbers behind them.
- Gate: fail-fast 3. If no layer separates base from filtered above the
  bootstrap spread, the candidate has no dynamic range on the easiest pair.
  Write the negative with the range estimate as the result.

## T6. Red-team: 1h [rev: was 1.5h]
- Description: once figures exist, list alternative explanations and test the
  plausible ones. Known candidates: activation-norm differences between
  checkpoints (probe learning scale not direction), control-set probe matching
  the target probe, train/held-out leakage, tokenization differences at the
  probe site.
- Use an anti-sycophancy prompt in a fresh session to generate the list.
- Deliverable: the list, with which explanations were tested and the outcome.
- Gate: fail-fast 4. If an artifact explains the separation, say so and report
  the corrected result.

## T7. Sanity checks documented: 1h
- Description: the verifications from Neel's evaluation tab. Weight hashes
  confirming which checkpoints loaded. One headline number recomputed by hand
  from raw outputs. Ten randomly selected prompts shown with their probe
  outputs. Ten reasoning entries from the cloze dataset read, two included.
- Deliverable: a short section in the write-up stating what was checked and how.
- Gate: none.

## T8. Logit lens: 1.5h (stretch, out by default)
- Only if T5 finishes with time to spare. Same cached activations, per-layer
  readout of the correct answer's rank. Supporting figure, not the headline.

## T9. Write-up: 2h counted + 2h uncounted (exec summary)
- Description: write-up body with one figure per experiment, methods from
  eval_spec.md and this plan, limitations paragraph. Then executive summary
  under 600 words in the 2 extra hours. Form questions after, uncounted
  (0.5-1h).
- [rev] Limitations to name: the suppressed label is unvalidated (no recovery
  arm); the claim is about linear decodability, not functional access;
  quantization recovery is the cheap next step that would connect the two;
  bootstrap spread is not training-run variance and there is no seed pair; CB
  was selected against the Cloze metric; the three checkpoints differ in three
  ways at once.
- Deliverable: the Google Doc, link-shareable, and the form.
- Gate: none. Own voice throughout.

## Buffer: 1h [rev: was 1.5h]

## Ledger
Tracked against ticket ids in the weekly thread.

## Deviations
- 2026-08-26, T1: MMLU pre-registered as a few-hundred-item subset. Ran the full 14042 instead. Reason: a full pass costs ~5 min per model, cheaper than estimated. Bio-adjacent subject exclusion applied post hoc from per-subject results as planned.
- 2026-08-26, tooling: Python env moved from the volume (.venv) to container disk(/root/venv), rebuilt by bootstrap on each pod. Reason: volume small-file latency stalled imports and the harness task index. Infra change, does not affect measurements.
- 2026-08-27, T1 actual vs estimate: estimated 2:00, actual about 3:00  including the results log and provenance items. Ratio 1.5x, matching the factor applied to T4 and T5.
- 2026-08-27, T2 actual: 0:45 against 1:00 estimate.
- 2026-08-27, T3 actual: 0:50 against 1:00 estimate.
- 2026-08-28, tooling: tested UV_CACHE_DIR on the volume to speed bootstrap.
  No gain (still ~30 min with a warm cache), the volume's small-file reads
  offset the download saving. Line removed from pod_bootstrap.sh. Conclusion:
  budget 30 min of pod bootstrap per deploy and do laptop work during it.
  A custom template image would remove the cost but is not worth building for
  the remaining pod-days of this window.