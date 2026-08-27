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
by intended forgetting. it moved every model by at most 0.2 points. The
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