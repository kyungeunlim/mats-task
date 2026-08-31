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
TODO after results. Pre-commitment: whichever outcome obtains, state the
strongest alternative reading and what was done to test it (T6 output). [YOU]

## Sanity checks
Condense from results.md checks plus T7 when done: harness cross-check,
independent MMLU recompute, TL-vs-HF residual check (max diff, tolerance),
position-token printout, weight hashes, N raw examples shown.

## Limitations
From plan.md T9 list: suppressed label unvalidated (no recovery arm), claim is
linear decodability not functional access, quantization recovery as the cheap
next step, bootstrap spread is not training-run variance, CB selected against
the Cloze metric, three checkpoints differ in three ways at once, control set
not perfectly neutral. TODO: prose version. [YOU]

## LLM usage
TODO: honest account. Structure exists: plan drafted with Claude and reviewed
adversarially in a fresh session (docs/plan_review_20260826.md), scripts
agent-written with prompts preserved in docstrings, key numbers verified by
independent recompute, design decisions made by me after questioning. [YOU]

## Time log
Toggl screenshot + ticket ledger from plan.md deviations.