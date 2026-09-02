"""T7 sanity check: show randomly selected held-out items with their probe
scores, and recompute argmax4 directly from the saved scores.

Implements the "ten randomly selected prompts shown with their probe
outputs" and "one headline number recomputed by hand from raw outputs"
items of docs/plan.md T7.

Prompt (2026-09-02):
    Write scripts/show_examples.py for the T7 sanity checks. It loads
    results/probe_heldout_scores.npz and data/main_set.json, takes a layer
    index and a model key, and prints N randomly selected held-out items
    with a stated seed. For each item: the question text, the four
    candidate texts, the four probe scores from the npz, which candidate
    the probe scored highest, which is correct, and whether they match.
    Print the seed and the selection method in the header so the
    selection is reproducible and visibly not cherry-picked. Default N to
    10. Also print the argmax4 accuracy over the full held-out set
    computed directly from the npz scores, so it can be compared against
    the value in results/probe_sweep.json as an independent recompute.
    Write the script only, do not run it.

The npz holds the held-out decision scores written by probe_bootstrap.py:
"scores_<set>_<model>" of shape (n_layers, n_heldout_items, 4), indexed in
the order of "heldout_ids_<set>", which are positions into the set file's
"items" list. The recompute here is argmax over axis -1 compared with each
item's "answer", written out without importing the metric code from
probe_one_layer.py.

Two reference values are printed next to the recompute. probe_bootstrap.json
"argmax4" is the point estimate from the same fit that produced the npz, so
it should match exactly. probe_sweep.json "argmax4" comes from a separate
lbfgs fit at the same C (the CV refit), so it can differ slightly in cells
where lbfgs did not converge; docs/results.md notes these are control-set
cells only.

Usage:
    python scripts/show_examples.py LAYER MODEL [--set main] [-n 10] [--seed 7]
        [--scores results/probe_heldout_scores.npz] [--sweep results/probe_sweep.json]
        [--bootstrap results/probe_bootstrap.json]
    MODEL is one of the npz keys' model suffixes: unfiltered,
    e2e-strong-filter, unlearned-cb.
"""

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SET_FILES = {"main": REPO_ROOT / "data" / "main_set.json",
             "control": REPO_ROOT / "data" / "control_set.json"}
LETTERS = "ABCD"
# Distinct from the split seed 42, shuffle seeds 1001/1002, and bootstrap
# seeds 2001/2002 used elsewhere. Overridable with --seed.
DEFAULT_SEED = 7


def lookup(records: list[dict], layer: int, set_name: str, model: str,
           key: str):
    for r in records:
        if r["layer"] == layer and r["set"] == set_name and r["model"] == model:
            return r.get(key)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Show random held-out items with probe scores and "
                    "recompute argmax4 from the saved scores (T7)")
    ap.add_argument("layer", type=int, help="layer index, 0-based")
    ap.add_argument("model",
                    help="model key as in the npz: unfiltered, "
                         "e2e-strong-filter, unlearned-cb")
    ap.add_argument("--set", dest="set_name", choices=sorted(SET_FILES),
                    default="main")
    ap.add_argument("-n", type=int, default=10,
                    help="number of held-out items to show (default 10)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"RNG seed for item selection (default {DEFAULT_SEED})")
    ap.add_argument("--scores", type=Path,
                    default=REPO_ROOT / "results" / "probe_heldout_scores.npz")
    ap.add_argument("--sweep", type=Path,
                    default=REPO_ROOT / "results" / "probe_sweep.json")
    ap.add_argument("--bootstrap", type=Path,
                    default=REPO_ROOT / "results" / "probe_bootstrap.json")
    ap.add_argument("--width", type=int, default=88,
                    help="wrap width for question and candidate text")
    args = ap.parse_args()

    z = np.load(args.scores)
    scores_key = f"scores_{args.set_name}_{args.model}"
    if scores_key not in z:
        available = sorted(k for k in z.keys() if k.startswith("scores_"))
        raise SystemExit(f"{scores_key} not in {args.scores}; "
                         f"available: {available}")
    all_scores = z[scores_key]          # (n_layers, n_ho, 4)
    heldout_ids = z[f"heldout_ids_{args.set_name}"]
    n_layers = all_scores.shape[0]
    if not 0 <= args.layer < n_layers:
        raise SystemExit(f"layer {args.layer} out of range 0..{n_layers - 1}")
    scores = all_scores[args.layer]     # (n_ho, 4)

    set_path = SET_FILES[args.set_name]
    d = json.loads(set_path.read_text())
    items = d["items"]
    # The npz's held-out order must match the set file's split, otherwise
    # scores and answers would be misaligned.
    assert list(heldout_ids) == list(d["heldout_indices"]), \
        "held-out ids in npz differ from set file heldout_indices"
    assert scores.shape == (len(heldout_ids), 4), scores.shape
    answers = np.array([items[i]["answer"] for i in heldout_ids])

    # ---- independent recompute over the full held-out set ----
    pred = scores.argmax(axis=-1)
    n_correct = int((pred == answers).sum())
    n_ho = len(heldout_ids)
    argmax4 = n_correct / n_ho

    sweep_val = boot_val = None
    if args.sweep.exists():
        sweep_val = lookup(json.loads(args.sweep.read_text())["records"],
                           args.layer, args.set_name, args.model, "argmax4")
    if args.bootstrap.exists():
        boot_val = lookup(json.loads(args.bootstrap.read_text())["heldout"],
                          args.layer, args.set_name, args.model, "argmax4")

    # ---- selection ----
    rng = np.random.default_rng(args.seed)
    n_show = min(args.n, n_ho)
    chosen = np.sort(rng.choice(n_ho, size=n_show, replace=False))

    # ---- header ----
    print("=" * args.width)
    print(f"T7 sanity check: held-out items with probe scores")
    print(f"scores file : {args.scores}")
    print(f"set file    : {set_path}")
    print(f"set / model / layer : {args.set_name} / {args.model} / {args.layer}")
    print(f"held-out items      : {n_ho} (split seed in set file: "
          f"{d.get('seed', d.get('split_seed', 'n/a'))})")
    print(f"selection : numpy.random.default_rng(seed={args.seed})"
          f".choice({n_ho}, size={n_show}, replace=False), positions into the "
          f"held-out order, shown sorted by position")
    print(f"scores    : logistic-regression decision_function at cand_end, "
          f"higher = more 'correct-candidate'-like; argmax within item")
    print("-" * args.width)
    print(f"argmax4 over full held-out set, recomputed from npz: "
          f"{n_correct}/{n_ho} = {argmax4:.6f}")
    print(f"  probe_bootstrap.json argmax4 (same fit as npz)   : "
          f"{'n/a' if boot_val is None else f'{boot_val:.6f}'}")
    print(f"  probe_sweep.json argmax4 (separate CV refit)     : "
          f"{'n/a' if sweep_val is None else f'{sweep_val:.6f}'}")
    if boot_val is not None and abs(boot_val - argmax4) > 1e-9:
        print("  WARNING: recompute differs from probe_bootstrap.json; "
              "these should be identical")
    if sweep_val is not None and abs(sweep_val - argmax4) > 0.005:
        print("  NOTE: recompute differs from probe_sweep.json by more than "
              "0.005 (separate lbfgs fit; see probe_bootstrap.py docstring)")
    print("=" * args.width)

    # ---- per-item display ----
    wrap = textwrap.TextWrapper(width=args.width, initial_indent="  ",
                                subsequent_indent="  ")
    cand_wrap = textwrap.TextWrapper(width=args.width, initial_indent="",
                                     subsequent_indent=" " * 18)
    n_match = 0
    for k, pos in enumerate(chosen, 1):
        item_id = int(heldout_ids[pos])
        it = items[item_id]
        s = scores[pos]
        p, a = int(pred[pos]), int(answers[pos])
        match = p == a
        n_match += match
        print(f"\n[{k}/{n_show}] held-out position {pos}, item index {item_id}")
        print("  Q:")
        print(wrap.fill(it["question"]))
        for c in range(4):
            tag = []
            if c == p:
                tag.append("probe top")
            if c == a:
                tag.append("correct")
            tag_s = f" <- {', '.join(tag)}" if tag else ""
            line = f"    {LETTERS[c]}  {s[c]:+8.3f}  {it['choices'][c]}{tag_s}"
            print(cand_wrap.fill(line))
        print(f"  probe top = {LETTERS[p]}, correct = {LETTERS[a]}, "
              f"match = {'YES' if match else 'no'}")

    print("\n" + "-" * args.width)
    print(f"shown items: {n_match}/{n_show} matched "
          f"(full held-out set: {n_correct}/{n_ho} = {argmax4:.4f})")


if __name__ == "__main__":
    main()
