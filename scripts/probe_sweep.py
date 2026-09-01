"""Full T5 probe sweep: every layer, both item sets, all three models.

Prompt (2026-09-01, implements plan.md T5, no bootstrap yet):
    Write scripts/probe_sweep.py, the full T5 run, reusing the fitting and
    metric code from probe_one_layer.py. For every layer 0 to 31, both item
    sets, and all three models: fit the probe per T2 and record argmax4, AUC,
    accuracy, the chosen C, and both shuffled baselines. Write results to a
    JSON or CSV under results/ so plotting and bootstrapping can read them
    without refitting. Print progress per layer. Suppress the sklearn
    FutureWarnings. No bootstrap yet, that comes separately. Do not run git
    commands.

Per (layer, set, model) this runs the three fits of probe_one_layer.py — the
real probe plus the two T2 shuffled-label baselines (within-item seed 1001
read on argmax4, global seed 1002 read on AUC) — and records argmax4, AUC,
thresholded accuracy, the CV-chosen C, both baseline readings, and the fit
wall time. 32 layers x 2 sets x 3 models x 3 fits = 576 CV fits, roughly one
to two hours; run as a background script with a log.

Output: one JSON (default results/probe_sweep.json) with a "meta" block
(set files and their seeds, shuffle seeds, C grid, fold count, layer dir,
model revisions from the layer-file meta) and a flat "records" list keyed by
layer/set/model, so plotting and bootstrapping read it without refitting.
The file is rewritten atomically after every layer as a checkpoint, so a
crash loses at most the layer in progress.

Usage:
    /root/venv/bin/python scripts/probe_sweep.py \
        [--layer-dir /workspace/cand_end_layers] [--out results/probe_sweep.json]
"""

import os

# Must precede the sklearn import (via probe_one_layer) and propagates to
# joblib worker processes, which the in-process warnings filter does not.
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")

import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from probe_one_layer import (CS_GRID, GLOBAL_SHUFFLE_SEED, N_FOLDS,
                             WITHIN_ITEM_SHUFFLE_SEED, argmax_acc,
                             build_examples, fit_probe)

warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent
SET_FILES = {"main": REPO_ROOT / "data" / "main_set.json",
             "control": REPO_ROOT / "data" / "control_set.json"}


def load_set(path: Path) -> dict:
    """Load a T3 item-set file: ids, answers, and its seed fields."""
    d = json.loads(path.read_text())
    train_ids, heldout_ids = d["train_indices"], d["heldout_indices"]
    assert not set(train_ids) & set(heldout_ids), f"{path}: split overlap"
    assert len(train_ids) + len(heldout_ids) == len(d["items"])
    answers = {i: it["answer"] for i, it in enumerate(d["items"])}
    return {
        "train_ids": train_ids,
        "heldout_ids": heldout_ids,
        "answers": answers,
        "answers_ho": np.array([answers[i] for i in heldout_ids]),
        "n_items": len(d["items"]),
        "seeds": {k: d[k] for k in d if k.endswith("seed")},
        "file": str(path.relative_to(REPO_ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="T5 probe sweep over all layers, both sets, three models")
    parser.add_argument(
        "--layer-dir", type=Path,
        default=Path("/workspace/cand_end_layers"),
        help="directory of layer_{L:02d}.pt files from extract_cand_end.py")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "results" / "probe_sweep.json",
        help="output JSON path (rewritten atomically after every layer)")
    args = parser.parse_args()

    sets = {name: load_set(path) for name, path in SET_FILES.items()}

    first = torch.load(args.layer_dir / "layer_00.pt")
    n_layers = first["meta"]["n_layers"]
    assert n_layers == 32, f"expected 32 layers, meta says {n_layers}"

    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "layer_dir": str(args.layer_dir),
        "n_layers": n_layers,
        "position_name": first["meta"]["position_name"],
        "models": first["meta"]["models"],
        "revisions": first["meta"]["revisions"],
        "item_sets": {name: {"file": s["file"], "seeds": s["seeds"],
                             "n_train_items": len(s["train_ids"]),
                             "n_heldout_items": len(s["heldout_ids"])}
                      for name, s in sets.items()},
        "cs_grid": [float(c) for c in CS_GRID],
        "n_folds": N_FOLDS,
        "cv_scoring": "roc_auc",
        "within_item_shuffle_seed": WITHIN_ITEM_SHUFFLE_SEED,
        "global_shuffle_seed": GLOBAL_SHUFFLE_SEED,
    }

    records = []
    t_start = time.time()
    for layer in range(n_layers):
        payload = first if layer == 0 else torch.load(
            args.layer_dir / f"layer_{layer:02d}.pt")
        assert payload["meta"]["layer"] == layer, "layer file/meta mismatch"
        for set_name, s in sets.items():
            assert payload["item_indices"][set_name] == list(range(s["n_items"])), (
                f"{set_name}: cached item indices are not the identity ordering")
            for model_key, acts in payload["acts"][set_name].items():
                X_tr, y_tr, g_tr = build_examples(acts, s["train_ids"],
                                                  s["answers"])
                X_ho, y_ho, _ = build_examples(acts, s["heldout_ids"],
                                               s["answers"])
                t0 = time.time()

                sc, chosen_c = fit_probe(X_tr, y_tr, g_tr, X_ho)
                rng_wi = np.random.default_rng(WITHIN_ITEM_SHUFFLE_SEED)
                y_wi = rng_wi.permuted(y_tr.reshape(-1, 4), axis=1).reshape(-1)
                sc_wi, _ = fit_probe(X_tr, y_wi, g_tr, X_ho)
                rng_gl = np.random.default_rng(GLOBAL_SHUFFLE_SEED)
                sc_gl, _ = fit_probe(X_tr, rng_gl.permutation(y_tr), g_tr, X_ho)

                rec = {
                    "layer": layer,
                    "set": set_name,
                    "model": model_key,
                    "argmax4": argmax_acc(sc, s["answers_ho"]),
                    "auc": float(roc_auc_score(y_ho, sc)),
                    "acc": float(np.mean((sc > 0) == y_ho)),
                    "chosen_C": chosen_c,
                    "shuf_argmax4": argmax_acc(sc_wi, s["answers_ho"]),
                    "shuf_auc": float(roc_auc_score(y_ho, sc_gl)),
                    "fit_seconds": round(time.time() - t0, 1),
                }
                records.append(rec)
                print(f"L{layer:02d} {set_name:<7} {rec['model']:<18} "
                      f"argmax4 {rec['argmax4']:.4f}  AUC {rec['auc']:.4f}  "
                      f"acc {rec['acc']:.4f}  shufA4 {rec['shuf_argmax4']:.4f}  "
                      f"shufAUC {rec['shuf_auc']:.4f}  C {rec['chosen_C']:.4g}  "
                      f"({rec['fit_seconds']:.0f}s)", flush=True)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"meta": meta, "records": records}, indent=1))
        tmp.replace(args.out)
        done = layer + 1
        elapsed = time.time() - t_start
        print(f"-- layer {layer} checkpointed to {args.out} "
              f"({done}/{n_layers} layers, {elapsed / 60:.1f} min elapsed, "
              f"~{elapsed / done * (n_layers - done) / 60:.0f} min left)",
              flush=True)

    print(f"\nDone: {len(records)} records in {args.out}")


if __name__ == "__main__":
    main()
