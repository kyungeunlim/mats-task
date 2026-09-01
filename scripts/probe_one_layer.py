"""Single-layer probe check before the full T5 pipeline (plan.md T5).

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

Cross-validation folds are grouped by item (GroupKFold on the item id), so
the four candidates of one item never straddle a fold boundary; otherwise
the CV score would leak within-item information. The shuffled-label baseline
permutes the training labels (fixed seed), refits the identical pipeline
including the CV over C, and scores against the true held-out labels; with a
1-in-4 positive rate its expected accuracy is the majority rate 0.75, which
is also printed.

Usage:
    /root/venv/bin/python scripts/probe_one_layer.py 16 /workspace/cand_end_layers
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold

MAIN_SET_FILE = Path(__file__).resolve().parent.parent / "data" / "main_set.json"
CS_GRID = np.logspace(-4, 2, 7)
N_FOLDS = 5
SHUFFLE_SEED = 0


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


def fit_and_score(X_tr, y_tr, groups_tr, X_ho, y_ho):
    """Standardize on X_tr, fit LogisticRegressionCV with item-grouped folds,
    return (held-out accuracy, chosen C)."""
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    X_tr = (X_tr - mu) / sd
    X_ho = (X_ho - mu) / sd
    splits = list(GroupKFold(n_splits=N_FOLDS).split(X_tr, y_tr, groups_tr))
    clf = LogisticRegressionCV(Cs=CS_GRID, cv=splits, scoring="accuracy",
                               solver="lbfgs", max_iter=1000, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    return clf.score(X_ho, y_ho), float(clf.C_[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-layer cand_end probe check on the main set (T5 pre-check)"
    )
    parser.add_argument("layer", type=int, help="layer index (0-based)")
    parser.add_argument(
        "layer_dir", type=Path, nargs="?",
        default=Path("/workspace/cand_end_layers"),
        help="directory of layer_{L:02d}.pt files from extract_cand_end.py")
    args = parser.parse_args()

    payload = torch.load(args.layer_dir / f"layer_{args.layer:02d}.pt")
    assert payload["meta"]["layer"] == args.layer, "layer file/meta mismatch"

    split = json.loads(MAIN_SET_FILE.read_text())
    train_ids, heldout_ids = split["train_indices"], split["heldout_indices"]
    assert not set(train_ids) & set(heldout_ids), "train/held-out overlap"
    assert len(train_ids) + len(heldout_ids) == len(split["items"])
    answers = {i: it["answer"] for i, it in enumerate(split["items"])}

    # Rows of the cached tensor are in item_indices order; the probe indexes
    # items by their global id, so the two orders must agree.
    assert payload["item_indices"]["main"] == list(range(len(split["items"]))), (
        "cached main-set item indices are not the identity ordering"
    )

    print(f"layer {args.layer}  ({args.layer_dir})")
    print(f"main set: {len(train_ids)} train items -> {4 * len(train_ids)} "
          f"examples, {len(heldout_ids)} held-out -> {4 * len(heldout_ids)} "
          f"examples (seed {split['seed']}, train_frac {split['train_frac']})")
    print(f"C grid {CS_GRID}, {N_FOLDS}-fold GroupKFold by item, "
          f"shuffle seed {SHUFFLE_SEED}\n")

    rng = np.random.default_rng(SHUFFLE_SEED)
    header = f"{'model':<20} {'held-out acc':>12} {'shuffled acc':>12} {'chosen C':>10}"
    print(header)
    for model_key, acts in payload["acts"]["main"].items():
        X_tr, y_tr, g_tr = build_examples(acts, train_ids, answers)
        X_ho, y_ho, _ = build_examples(acts, heldout_ids, answers)
        t0 = time.time()
        acc, c_real = fit_and_score(X_tr, y_tr, g_tr, X_ho, y_ho)
        acc_shuf, _ = fit_and_score(X_tr, rng.permutation(y_tr), g_tr, X_ho, y_ho)
        print(f"{model_key:<20} {acc:>12.4f} {acc_shuf:>12.4f} {c_real:>10.4g}"
              f"   ({time.time() - t0:.0f}s)", flush=True)
    print(f"\nmajority rate on held-out (always predict wrong-candidate): "
          f"{1 - np.mean([answers[i] == c for i in heldout_ids for c in range(4)]):.4f}")


if __name__ == "__main__":
    main()
