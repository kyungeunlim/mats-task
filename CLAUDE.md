# MATS 12.0 Application Task

## Project
Mechanistic interpretability research task for Neel Nanda's MATS stream.
16-20 hour window (excluding setup and general learning), +2 hours for
executive summary only. Working notes and write-up drafted as the work
proceeds, not at the end.

## Reference material
- docs/mech_interp_context.md: ~600k tokens of concatenated mech interp
  reference (TransformerLens, nnsight source/docs, ARENA tutorials, key
  papers, Neel Nanda's posts). NEVER read this file whole. Grep/search it
  for relevant sections when an interpretability question comes up,
  instead of answering from memory.

## Environment

Experiments run on a RunPod pod (A100 80GB, US-KS-2), volume at /workspace.
Writing and light edits happen on the laptop; the repo is the sync point, so commit and push often.
Pod restart procedure: POD_SETUP.md. Run it after every pod start; container disk is wiped on stop.

- Repo: /workspace/mats-task on the pod, ~/Projects/mats-task on the laptop
- Python: uv venv, Python 3.12. On the pod it is /root/venv (container disk,
  rebuilt by pod_bootstrap.sh). On the laptop it is .venv in this repo. Install
  with `uv pip install --python <that venv>/bin/python <pkg>`. Never install
  into system Python or conda base.
- TransformerLens v3 emits a deprecation warning on
  HookedTransformer.from_pretrained; the old path works fine. Do not
  migrate to TransformerBridge unless TL becomes load-bearing for the
  actual task.
- HF_HOME=/workspace/hf (pod). Three checkpoints already cached: deep-ignorance-unfiltered,
  deep-ignorance-e2e-strong-filter, kelim/deep-ignorance-unlearned-cb. Do not re-download.
- Datasets cached: cais/wmdp (wmdp-bio, 1273), cais/mmlu (all, 14042 test)
- First imports from the volume take 30-90s. Normal, not a hang.
- Save plots as PNGs under results/

## Working rules
- When proposing file edits, always present the complete diff in the
  message body before applying, never a truncated preview. I review
  every diff before approving.
- Persistent state: load models/data once in dedicated cells at the top
  of the Jupyter kernel. Never restart the kernel without asking.
- Save every plot to disk as PNG (results/) in addition to displaying it.
- Checkpoint expensive artifacts (activations, datasets) to disk so a
  crashed kernel is not a disaster. Long jobs run as background scripts
  with logs, not notebook cells.
- Verification before claiming done: for any key number, show the code
  path that produced it and print the raw value. Flag any result that
  depends on an unverified assumption.
- When asked to write a report of what was done, include concrete
  technical detail: exact prompts, hyperparameters, shapes, file paths.

## Style
- Plain, factual register. No superlatives. Scoped claims with honest
  hedging. State what failed as plainly as what worked.

## Do not
- Do not edit files under docs/ (reference material).
- Do not commit large binaries (*.pt, activations, results/) — see
  .gitignore.