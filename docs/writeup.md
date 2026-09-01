# Suppression vs removal: probing what an unlearning fine-tune left readable
(working title, TODO)

Status: draft in progress. Final version goes to a single-tab Google Doc,
anyone-with-link, before submission.

## Executive summary (under 600 words, one graph per key experiment)
TODO last. [YOU]

## Question
TODO: one paragraph. What I asked and why it is not answerable from behavior
alone. Base material: plan.md T2 framing sentence, scoping doc section I. [YOU]

## Setup
Three checkpoints from one lineage (DI base, e2e-strong-filter, no-LoRA CB),
what each is, why this triple lets the question be asked.
TODO: two paragraphs. Base material: results.md header, eval_spec.md purpose. [YOU]
- Checkpoint revisions: paste the three hashes from results.md.
- Tooling: lm-eval 0.4.12 for behavior, TransformerLens 3.7.3 for activations
  (verified against a direct HF forward pass before caching, see Sanity checks).
  Activations were cached under Python 3.12.3, torch 2.13.0+cu130, on an
  RTX PRO 6000 (driver CUDA 13.2).  


## Behavioral anchor
Paste the two tables from results.md (Cloze, MMLU with all three columns), then:
TODO: the reading, condensed from results.md interpretation paragraphs. [YOU]
- Includes: replication vs selection-metric caveat, the spillover observation
  (filter cost general biology, fine-tune did not).

## Probe design (pre-registered)
Condense from plan.md T2: target, formulation, positions, split, controls,
what each outcome would mean, and the shape-not-level note.
TODO: half a page. This section shows the design was fixed before results. [YOU]

## Results
TODO: the probe figure(s) and what they show. Cannot exist before T5.
- Figure: accuracy vs layer, three models, error bands, chance line,
  intervention layers marked.
- Control set comparison.


## What would change my mind / strongest evidence against

NOTES, NOT PROSE [rewrite before submission]

The pre-commitment in T2 was: whichever outcome obtains, state the
strongest alternative reading and what was done to test it. The
complication is that T2's outcome list assumed the two known states
would calibrate a scale, and they did not, so the outcomes as written
do not map onto what happened.

Which outcome obtained. CB tracks base at every layer, which is T2's
outcome (a). T2 already called this weak, since CB was fine-tuned from
base while filtered was pretrained separately, so CB resembling base was
close to guaranteed by construction. The informative outcome (c), CB
tracking base early and falling away at or after the intervention
layers, did not appear.

But the reading T2 assigned to (a), that information about the answer is
still present, does not follow here. Filtered reads 0.42 to 0.48 on the
same metric, and filtered never saw the forget corpus. A high probe
score therefore does not indicate retained knowledge, because the same
score appears where there is nothing to retain. So CB reading high says
nothing about CB.

The strongest reading the data supports: the probe does not separate the
two known states, so it cannot place CB between them. That is a
statement about the instrument, not about the intervention.

Alternative explanations for the probe reading high on all three models,
none tested here:
- The probe reads question-candidate plausibility rather than the
  specific knowledge. The control set was built to test this and gives
  a partial answer, but the control is itself affected by the filter
  (see Limitations), so it does not settle it.
- The probe reads general biology competence that survives filtering.
- The signal comes from the candidate text alone rather than
  question-candidate fit. Testable by fitting a probe on question-free
  prompts; not run.

What would change my mind about the negative reading:
- A removal reference whose probe score sits at chance. That would
  restore the floor and make the scale usable. Requires a better
  characterized model organism, not a different metric.
- A seed pair, so the noise floor is measured rather than approximated
  by resampling a fixed model.
- Recovery experiments. Quantization recovery is the cheap version:
  quantize each model, re-run Cloze, and see whether CB's score rises
  toward base while filtered's does not. That asks the suppression
  question behaviorally and sidesteps the linear-decodability
  limitation. Not run; the obvious next step.


## Sanity checks
Condense from results.md checks plus T7 when done: harness cross-check,
independent MMLU recompute, TL-vs-HF residual check (max diff, tolerance),
position-token printout, weight hashes, N raw examples shown.


"grep -c WARNING results/eval/probe_bootstrap_20260901.log
grep WARNING results/eval/probe_bootstrap_20260901.log | grep -c main
10
0" 

Clean result: 10 warnings, none on the main set. So every fixed-C refit on main reproduced the sweep's point estimate to within 0.005, and the divergences are confined to control cells, which is where CV chose weak regularization and lbfgs hit the iteration limit.

That's worth one line in the sanity-check section, since it's a free cross-check that the bootstrap's refits match what the sweep produced.


## Limitations

NOTES, NOT PROSE [rewrite before submission]

From plan.md T9:
- The suppressed label is unvalidated. No recovery arm, so "suppressed" is a
  reading, not a demonstration.
- The claim is linear decodability, not functional access.
- Quantization recovery is the cheap next step that would connect the two.
- Bootstrap spread is not training-run variance. No seed pair.
- CB was selected against the Cloze metric, so its T1 number is a replication
  of its own selection target.
- The three checkpoints differ in three ways at once.
- The control set is not perfectly neutral: filtered scored about 4.7 points
  below base on the eight MMLU bio/med subjects (T1).

Added 2026-09-01:
- Linear probe only. A negative result is weak: it says a linear reader cannot
  find the information, not that the information is absent. A nonlinear probe
  might find it. Linear was chosen because a high-capacity probe can construct
  the answer from a rich representation the model itself does not use.
- A positive result does not show the model can use the information. The probe
  is an external reader. Presence in the residual stream is necessary for
  recovery but not sufficient.
- One position only. cand_end carries the label under the T2 formulation;
  q_end and answer_marker were cached but not probed.
- The residual stream is one of several possible targets. Attention head
  outputs or MLP outputs could localize the effect, and would be the natural
  follow-up given that the ERA finding on the tested CB+LAT variant was
  confined to QKV weights.
- Metric changed from accuracy after the pre-registration was written, for a
  reason that follows from the label structure rather than from results. See
  plan.md Deviations 2026-09-01.
- Convergence: lbfgs hit the 1000-iteration limit on many fits, concentrated at
  the weaker regularization settings. Fits are consistent across models and
  layers so the comparison holds, but the coefficients are not converged.
- Verification coverage: the TL-vs-HF check ran on one prompt, the batch
  padding check on four items. Position finding and NaN checks covered every
  sequence.
- Layer indices are absolute and the three checkpoints share an architecture,
  so the comparison is direct here. Transferring the method across
  architectures would need a depth convention, and the more portable framing is
  a fine-tune measured against its own base rather than an absolute layer.


Added 2026-09-01, from the T5 result:

- The removal reference is not calibrated. e2e-strong-filter was
  intended as the "knowledge removed" endpoint, but it reads 0.42 to
  0.48 on main-set argmax while scoring 0.2435 on Cloze. The behavioral
  floor is at chance; the probe floor is not. With no floor, a third
  model cannot be placed between the endpoints, so the design cannot
  locate CB on a removed-to-intact scale.

- The removal reference is also not clean. Per the Deep Ignorance paper,
  the blocklist was extracted from the WMDP Forget papers and reports
  58.28% precision on the WMDP-only eval set and 44.02% with
  Llama-labeled DCLM documents included. e2e-strong-filter is
  blocklist-only with no classifier review, removing 8.42% of
  pretraining documents, and the authors state they erred toward
  over-filtering. Appendix J states they did not measure filter
  performance on the actual training data and expect high false-positive
  rates, with sampled removals including general biomedical papers. So
  "filtered" means a model missing an imprecisely specified corpus
  slice, roughly half of it removed in error, not a model with a known
  body of knowledge absent.

- Dynamic range is comparable to noise. Base minus filtered on main-set
  argmax is 0.04 to 0.06 in the mid-to-late layers, against an item-axis
  95% half-width of about 0.024 per point and a fit-axis standard
  deviation of 0.015 to 0.024 per point. No single layer resolves the
  separation. The sign is consistent from about layer 13 onward, which
  is suggestive, but layers are not independent so that is not a test.

- Probe accuracy is not a measure of how the internals differ. It is one
  projection of a 4096-dimensional difference onto an axis selected by
  the label. Two models could differ substantially in representation
  while correctness remains equally readable in both, or barely differ
  while readability differs. What is compared across models is a scalar
  summary in each model's own frame, not the representations themselves.

- The label is behavioral. The probe target is the benchmark answer key,
  so the instrument is coupled to the evaluation it was meant to be
  independent of.

- Probes can be trained against. Bailey et al. (arXiv 2412.09565) show
  representations can be shaped so a probe fails while behavior
  persists. A low probe score therefore has at least three explanations:
  the information is absent, it is present but not linearly readable, or
  the representation was shaped so this probe misses it. The T2 outcome
  list covers the first two.

- The bootstrap axes are not training-run variance. Both hold the
  trained models fixed, so the reported spread is a lower bound on the
  uncertainty that matters, and there is no seed pair.

- Convergence: lbfgs hit the 1000-iteration limit on many fits,
  concentrated in the weakly regularized control-set fits where CV found
  little to prefer. Fits are consistent across models and layers so the
  comparison holds, but the coefficients are not converged.

- Verification coverage: the TL-vs-HF check ran on one prompt, the batch
  padding check on four items. Position finding and NaN checks covered
  every sequence.



## LLM usage
TODO: honest account. Structure exists: plan drafted with Claude and reviewed
adversarially in a fresh session (docs/plan_review_20260826.md), scripts
agent-written with prompts preserved in docstrings, key numbers verified by
independent recompute, design decisions made by me after questioning. [YOU]

## Time log
Toggl screenshot + ticket ledger from plan.md deviations.


