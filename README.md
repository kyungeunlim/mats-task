# Pilot 1: a per-layer linear probe as a suppression-versus-removal detector

Sep 2026. One pilot in a project on internal metrics for knowledge suppression
versus removal.

**The program.** Safety benchmarks score what a model outputs, so they cannot
tell whether hazardous knowledge is gone or merely unused. The project asks
whether any internal quantity, derived from weights, activations or gradients,
is reliably sensitive to that difference. The framing is detector
characterization against calibration sources: models in known knowledge states
anchor the scale, and a candidate quantity is the detector reading. Two anchors
define it. Same-state variation is the null, how much the quantity moves between
two models that differ in nothing meaningful. The base-versus-filtered
separation is the largest response a genuine change in knowledge state is
expected to produce. A candidate is worth pursuing only if the second stands
clearly above the first, and the suppressed model is then placed on the scale
those anchors define.

**This pilot** takes one candidate from that pool, a per-layer linear probe on
the residual stream, and runs it end to end on the Deep Ignorance suite. It was
also submitted as a MATS 12.0 application task, which is where the time budget
visible in the plan and ledger comes from.

**Result: the probe could not do it here, and the obstacle sits upstream of the
probe.** The removal reference does not read low, so there is no floor and too
little span to place a third model within. The one separation that does resolve
also appears on general biology the filter never targeted, so it is not evidence
about the targeted knowledge.

## Setup

Three checkpoints from the Deep Ignorance suite, all 6.9B, sharing a training
recipe, tokenizer and hyperparameters. Each stands in for one knowledge state.

| state | model | checkpoint |
|---|---|---|
| retained | base | `EleutherAI/deep-ignorance-unfiltered` |
| removed | filtered | `EleutherAI/deep-ignorance-e2e-strong-filter` |
| suppressed | fine-tune | `kelim/deep-ignorance-unlearned-cb` |

The filtered model was pretrained on a corpus with biothreat material removed, so
it never learned it. That makes it a removal reference by construction, and the
reason this suite was chosen. The fine-tune applies circuit breakers to base
without LoRA; it is my own release. On WMDP-Bio Verified Cloze the filtered model
and the fine-tune both sit at chance while base does not, so the benchmark cannot
tell them apart. That is the premise the probe was built to look behind.

The probe is a logistic regression fit per layer, reading the residual stream at
the last token of a question-and-candidate pair and predicting whether that
candidate is correct. It is scored by within-item argmax over the four
candidates, which matches how the benchmark scores, with AUC alongside.
Uncertainty comes from a paired bootstrap over held-out items. A control set of
bio-adjacent MMLU items in the same prompt shape tests whether any separation is
about the targeted knowledge or about biology in general.

## What was found

![retained minus removed, and retained minus suppressed, on AUC](results/figures/probe/gap_auc.png)

- **The probe reads well above what the models output**, on every model. At layer
  20 the filtered model reads 0.458 on within-item argmax while scoring 0.2435
  behaviorally.
- **But the removal reference does not read low.** A high probe reading appears
  on the model built to lack the material, so it cannot indicate retained
  knowledge.
- **Retained versus removed resolves, and so does the same gap on general
  biology.** The left panel above shows both the main set and the control
  clearing zero from about layer 15. The design pre-committed that this pattern
  undermines the knowledge reading rather than supporting it.
- **Retained versus suppressed is mostly unresolved.** Three explanations remain
  open, and this design cannot separate them: the fine-tune changed little the
  probe reads, the instrument lacks resolution, or the instrument is structurally
  blind to what the circuit-breaker objective does.

## What this means for the program

The pilot did not reach the decision gate, and that is the useful part.

**The obstacle is an anchor, not a candidate.** The sensitivity criterion
compares a candidate's base-versus-filtered separation against same-state
variation. Here the separation itself turned out not to be interpretable, because
the removal reference reads high and reads high on untargeted biology too. Every
other candidate in the pool uses the same filtered model as its removal
reference, so this constrains weight spectra, representation similarity and logit
lens before any of them is run. Fixing it is upstream work on the anchors, not
work on the next candidate.

**Two things this pilot did not do**, worth stating so the result is not read as
more than it is.

- **No same-state estimate.** Both bootstrap axes here, held-out items and
  probe-training subsamples, hold the trained models fixed, so neither is
  training-run variance. Without a null, the probe's numbers cannot be expressed
  in units of same-state spread, and the candidate has not formally passed or
  failed the sensitivity criterion. It was stopped earlier than that.
- **The suppressed label is unvalidated.** No recovery attack was run, so the
  fine-tune's label rests on its construction rather than on demonstrated
  recoverability.

**One thing to carry forward into how candidates get characterized.** A
topic-matched control input set caught something the main set alone would not
have: the separation is not specific to the domain the filter targeted. For any
input-dependent candidate, a control set belongs in the characterization
alongside the target set, not as an optional extra.

## Repository

```
docs/plan.md        execution plan and deviations ledger
docs/results.md     results log, written as the work proceeded
docs/eval_spec.md   how the behavioral benchmarks were measured
data/               the two item sets with their splits
scripts/            one script per pipeline step
lm_eval_tasks/      vendored WMDP-Bio Verified Cloze task config
results/            benchmark outputs, run logs, figures
```

Two documents sit outside the repo. The **pilot report** is the account of record
for everything here, and the **scoping document** defines the program: the
knowledge-state definitions, the reliability criteria, and the pool of candidate
internal quantities this pilot drew from.

- Pilot report: https://docs.google.com/document/d/1Xdqm2_jXgoRPpiHp_25lvz7ATDddy1tUOT8MZaU-4LE
- Scoping document: https://docs.google.com/document/d/1JeIDDGg63wHvzlFoA5TDPKt7icVu4oHTR3vnr1E_cdU

The documents under `docs/` are working records written as the work proceeded:
the plan fixes the design in advance, and the results log records what each run
produced. Where they disagree with the report, the report is right. Earlier
write-up drafts are not included, having been replaced by the report.

Each script carries the prompt that produced it, the date, and the plan ticket it
implements. The pipeline runs in the order: build the prompt sets, verify
TransformerLens against a direct HF forward pass, cache activations, extract the
probe position, sweep all layers, bootstrap, then plot. Each script's docstring
carries its own usage.

Activation caching needs a 40GB+ GPU and writes tens of gigabytes, so the cache
is not in the repo. The held-out probe scores are, in
`results/probe_heldout_scores.npz`, so the probe analysis and the figures
reproduce without a GPU.

## Environment

Python 3.12, dependencies pinned in `requirements.txt`; torch 2.13.0,
transformer-lens 3.7.3, transformers 5.15.1, lm-eval 0.4.12, scikit-learn 1.9.0.
Activations were cached on an RTX PRO 6000.

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

`POD_SETUP.md` and the `pod_*.sh` scripts are the rented-GPU setup and are not
needed to read the results.

## Caveats

The probe shows information is readable, not that the model uses it, and
"suppressed" means present but unused, so readability is necessary evidence for
it and not sufficient. The control set is my reformatting of a different
benchmark, so it carries a judgement call the main set does not. The conclusion
is about one configuration: one target, one token position, one model family, one
comparison, so whether the reference reads high because of the model or because
of this probe cannot be settled from one probe alone. Sanity checks and the full
limitations are in the report and in `docs/results.md` under T7.

`docs/mech_interp_context.md` is third-party reference material gathered for this
task, not my writing.
