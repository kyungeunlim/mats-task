"""Single-layer probe check before the full T5 pipeline (plan.md T5; metric
and baselines per plan.md T2 as revised 2026-09-01).

Prompt (2026-09-01, pre-check for the plan.md T5 ticket):
    Write scripts/probe_one_layer.py, a minimal check of whether the T2 probe
    target separates the models at all, before the full T5 pipeline. Take a
    layer index and load the corresponding file from /workspace/cand_end_layers.
    For the main set, build the probe examples per T2: each item contributes
    four examples, one per candidate, labelled by whether that candidate is
    the correct answer. Use the item-level 70/30 split from the T3 id lists
    in data/, not a fresh split. Standardize features using means and standard
    deviations from the training items of that model only. Fit logistic
    regression with regularization strength chosen by cross-validation inside
    the training items. Report held-out accuracy for each of the three models,
    alongside the shuffled-label accuracy as the chance level. Print the
    numbers, no plotting. Take the layer index and the layer-file directory
    as arguments.

Prompt (2026-09-01, metric change, implements the revised "Metric" and
"Shuffled-label baselines" decisions in plan.md T2):
    Update scripts/probe_one_layer.py for the metric change now in docs/plan.md
    T2, which I've just pushed. Read that section first. Report three numbers
    per model: within-item argmax over an item's four candidates (chance 0.25),
    AUC over held-out examples, and accuracy, keeping accuracy only so the
    degenerate behavior stays visible. Score the cross-validation for C by AUC
    rather than accuracy. Add both shuffled-label baselines from T2: a
    within-item shuffle seeded 1001, read on argmax, and a global shuffle
    seeded 1002, read on AUC. Both permute labels only, leaving the vectors,
    the split, the class proportion, and the fitting procedure unchanged.
    Print the seeds alongside the results. Accept multiple layer indices on
    the command line so a few layers can be checked in one run. Do not run
    git commands.

Cross-validation folds are grouped by item (GroupKFold on the item id), so
the four candidates of one item never straddle a fold boundary; otherwise
the CV score would leak within-item information. Both baselines refit the
identical pipeline (standardization, grouped CV over C scored by AUC) on
permuted training labels and are scored against the true held-out labels.
The within-item shuffle re-draws which of each item's four candidates is
marked correct, preserving one positive per item so argmax stays
well-defined; the global shuffle permutes labels across all training
examples and is read on AUC only, per T2.

Usage:
    /root/venv/bin/python scripts/probe_one_layer.py 8 16 24 [--layer-dir /workspace/cand_end_layers]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

MAIN_SET_FILE = Path(__file__).resolve().parent.parent / "data" / "main_set.json"
CS_GRID = np.logspace(-4, 2, 7)
N_FOLDS = 5
WITHIN_ITEM_SHUFFLE_SEED = 1001  # baseline for argmax, plan.md T2
GLOBAL_SHUFFLE_SEED = 1002       # baseline for AUC, plan.md T2


def build_examples(acts: torch.Tensor, item_ids: list[int],
                   answers: dict[int, int]):
    """Flatten (items, 4, d_model) into per-candidate examples for item_ids.

    Returns X (4*len(item_ids), d_model) float32, y (correct-candidate flag),
    groups (item id per example, for grouped CV).
    """
    rows = acts[item_ids].to(torch.float32).numpy().reshape(-1, acts.shape[2])
    y = np.array([c == answers[i] for i in item_ids for c in range(4)])
    groups = np.repeat(item_ids, 4)
    return rows, y, groups


def fit_probe(X_tr, y_tr, groups_tr, X_ho):
    """Standardize on X_tr, fit LogisticRegressionCV (AUC-scored, item-grouped
    folds), return (held-out decision scores, chosen C)."""
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    splits = list(GroupKFold(n_splits=N_FOLDS).split(X_tr, y_tr, groups_tr))
    clf = LogisticRegressionCV(Cs=CS_GRID, cv=splits, scoring="roc_auc",
                               solver="lbfgs", max_iter=1000, n_jobs=-1)
    clf.fit((X_tr - mu) / sd, y_tr)
    return clf.decision_function((X_ho - mu) / sd), float(clf.C_[0])


def argmax_acc(scores, answers_ho):
    """Within-item argmax accuracy: fraction of held-out items whose true
    answer gets the highest probe score among the item's four candidates."""
    return float(np.mean(scores.reshape(-1, 4).argmax(axis=1) == answers_ho))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-layer cand_end probe check on the main set "
                    "(T5 pre-check, T2 metrics)"
    )
    parser.add_argument("layers", type=int, nargs="+",
                        help="layer indices (0-based), one or more")
    parser.add_argument(
        "--layer-dir", type=Path,
        default=Path("/workspace/cand_end_layers"),
        help="directory of layer_{L:02d}.pt files from extract_cand_end.py")
    args = parser.parse_args()

    split = json.loads(MAIN_SET_FILE.read_text())
    train_ids, heldout_ids = split["train_indices"], split["heldout_indices"]
    assert not set(train_ids) & set(heldout_ids), "train/held-out overlap"
    assert len(train_ids) + len(heldout_ids) == len(split["items"])
    answers = {i: it["answer"] for i, it in enumerate(split["items"])}
    answers_ho = np.array([answers[i] for i in heldout_ids])

    print(f"main set: {len(train_ids)} train items -> {4 * len(train_ids)} "
          f"examples, {len(heldout_ids)} held-out -> {4 * len(heldout_ids)} "
          f"examples (split seed {split['seed']}, train_frac "
          f"{split['train_frac']})")
    print(f"C grid {CS_GRID}, {N_FOLDS}-fold GroupKFold by item, CV scored "
          f"by AUC")
    print(f"chance: argmax4 0.25, AUC 0.5; accuracy kept only to show the "
          f"degenerate 0.75 majority behavior")
    print(f"shuffled-label baselines: within-item seed "
          f"{WITHIN_ITEM_SHUFFLE_SEED} (read on argmax4), global seed "
          f"{GLOBAL_SHUFFLE_SEED} (read on AUC); labels permuted, vectors/"
          f"split/class proportion/fitting unchanged")

    header = (f"{'model':<20} {'argmax4':>8} {'AUC':>7} {'acc':>7} "
              f"{'shuf-argmax4':>13} {'shuf-AUC':>9} {'chosen C':>9}")
    for layer in args.layers:
        payload = torch.load(args.layer_dir / f"layer_{layer:02d}.pt")
        assert payload["meta"]["layer"] == layer, "layer file/meta mismatch"
        assert payload["item_indices"]["main"] == list(range(len(split["items"]))), (
            "cached main-set item indices are not the identity ordering"
        )
        print(f"\nlayer {layer}  ({args.layer_dir})")
        print(header)
        for model_key, acts in payload["acts"]["main"].items():
            X_tr, y_tr, g_tr = build_examples(acts, train_ids, answers)
            X_ho, y_ho, _ = build_examples(acts, heldout_ids, answers)
            t0 = time.time()

            s, c_real = fit_probe(X_tr, y_tr, g_tr, X_ho)
            am = argmax_acc(s, answers_ho)
            auc = roc_auc_score(y_ho, s)
            acc = float(np.mean((s > 0) == y_ho))

            # Within-item shuffle: re-draw the correct candidate per training
            # item (permuting an item's one-hot labels equals choosing a
            # random position), keeping one positive per item.
            rng_wi = np.random.default_rng(WITHIN_ITEM_SHUFFLE_SEED)
            y_wi = rng_wi.permuted(y_tr.reshape(-1, 4), axis=1).reshape(-1)
            s_wi, _ = fit_probe(X_tr, y_wi, g_tr, X_ho)
            shuf_am = argmax_acc(s_wi, answers_ho)

            rng_gl = np.random.default_rng(GLOBAL_SHUFFLE_SEED)
            s_gl, _ = fit_probe(X_tr, rng_gl.permutation(y_tr), g_tr, X_ho)
            shuf_auc = roc_auc_score(y_ho, s_gl)

            print(f"{model_key:<20} {am:>8.4f} {auc:>7.4f} {acc:>7.4f} "
                  f"{shuf_am:>13.4f} {shuf_auc:>9.4f} {c_real:>9.4g}"
                  f"   ({time.time() - t0:.0f}s)", flush=True)

    print(f"\nseeds: split {split['seed']}, within-item shuffle "
          f"{WITHIN_ITEM_SHUFFLE_SEED}, global shuffle {GLOBAL_SHUFFLE_SEED}")
    print(f"majority rate on held-out (always predict wrong-candidate): "
          f"{1 - np.mean([answers[i] == c for i in heldout_ids for c in range(4)]):.4f}")


if __name__ == "__main__":
    main()
