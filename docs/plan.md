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
one tool, so consistency matters more than exact fidelity to the HF forward pass.

[rev] All three models are loaded with `from_pretrained_no_processing`. The default
`from_pretrained` applies `center_writing_weights` and `fold_ln`, which are
per-checkpoint transforms on the residual stream being compared. Verified in T4 by
checking one prompt's hidden state against an HF forward pass before caching.

Fallback if TL hits a memory or conversion problem on the L40: raw forward hooks on
the HF model, about 15 lines, not a migration.

---

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

## T2. Probe target spec and pre-registration: 1h [rev: was 0.5h]
- Description: written decisions, no code. All of the following are fixed before
  any activations are cached.
- Probe target: predict correct vs distractor completion from the residual state.
- [rev] Probe site: primary is the last question token (the token before
  "\nAnswer:"), where retrieval would occur before circuit-breaker rerouting
  acts. Secondary is the answer position. The contrast between the two sites is
  itself a result: readable upstream and absent downstream is the
  suppression-shaped pattern. A probe at the answer position alone is close to a
  restatement of the logits.
- [rev] Items: all 1076 cloze_compatible items, not the base-correct subset.
  Conditioning on base behavior would induce selection.
- [rev] Position 0 is excluded from every position-averaged quantity, decided now
  on the basis of the smoke test, not after inspecting this data.
- [rev] All layers are reported. No single layer is selected post hoc as "the"
  layer.
- [rev] Power: held-out n of roughly 300 gives a binomial SE near 3 points, so
  95% intervals of about ±6. Gaps between curves smaller than that are not
  resolvable by this design and will not be claimed.
- [rev] Regularization: probe C chosen by cross-validation within the training
  split only. Never on held-out.
- [rev] Pre-registered outcomes and what each licenses:
  (a) CB tracks base at every layer: the fine-tune barely moved the residual
      stream. Target information remains linearly decodable. Says little about
      functional access.
  (b) CB tracks filtered at every layer: target information is not linearly
      readable at this granularity. Consistent with removal or with suppression
      that is not linear at these sites.
  (c) CB layer-dependent or site-dependent, readable upstream and not at the
      answer site: the pattern most consistent with suppression as rerouting.
      The most informative outcome, and the one that would motivate the
      recovery experiment.
- Deliverable: this section filled in, plus which layers (target: every 2nd
  layer, all 32 if time allows).
- Gate: fail-fast 2. If the target would plausibly fire equally on all three
  models (measuring topic, not knowledge), redefine it.

## T3. Prompt sets: 1h
- Description: probe subset drawn from the 1076 cloze_compatible items with a
  logged seed, split into train/held-out. A topic-matched control set (to be
  decided in T2: same items with shuffled answers, or biology items without the
  target knowledge).
- Deliverable: two item-id lists committed to the repo, with the seed.
- Gate: none. Sanity check: item counts, no overlap between train and held-out.

## T4. Activation caching: 3h [rev: was 2h, 1.5x applied]
- Description: one forward pass per prompt per checkpoint, residual stream at
  the chosen layers and both probe sites, saved to disk. One model at a time on
  the L40.
- Start with base: verify TL hook names and tensor shapes on the 7B NeoX
  architecture.
- [rev] Before caching: load with `from_pretrained_no_processing`, run one fixed
  prompt through TL and through the HF model directly, confirm the final-layer
  residual matches to numerical tolerance. Log the flags used.
- Per-position norm check per model (position-0 exclusion is already decided,
  this confirms the shape).
- Deliverable: cached tensors on the volume, the script, a per-layer norm plot
  per model, the HF-vs-TL check output.
- Gate: none formal. Sanity: shapes match expectation, norms look sane, no NaNs.

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