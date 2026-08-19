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

## Working rules
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