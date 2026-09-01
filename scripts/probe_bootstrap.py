"""Bootstrap uncertainty for the T5 probe curves, per plan.md T2/T5.

Implements the [rev] error-bar deliverable of docs/plan.md T5: two bootstrap
axes, held-out items (item variance) and probe-training subsamples (fit
variance). Neither is training-run variance; that stays in limitations.

Prompt (2026-09-01):
    Write scripts/probe_bootstrap.py, adding uncertainty to the T5 curves per
    plan.md T2's two axes. Reuse the fitting and metric code from
    probe_one_layer.py.

    Axis one, held-out items, for all 32 layers, both sets, all three models:
    fit the probe once at the C already chosen in results/probe_sweep.json
    rather than redoing the CV, keep the held-out decision scores, then
    resample the held-out items with replacement 1000 times and recompute
    argmax4 and AUC each time. Resample at the item level so an item's four
    candidates move together, matching the split. Report the 2.5 and 97.5
    percentiles for each metric. Save the held-out decision scores to disk as
    well, so later analysis does not need to refit.

    Axis two, training subsamples, for a list of layers given by a
    --train-layers flag defaulting to 0, 15 and 31, main set only, three
    models: resample the training items with replacement, refit at the same
    fixed C, and recompute both metrics on the fixed held-out set. Default 30
    resamples, and report the standard deviation across them rather than
    percentiles, since 30 is too few for a stable tail percentile.

    Take the resample counts as flags. Use seeds distinct from 42, 1001 and
    1002, record them in the output, and write results to JSON alongside the
    sweep output. Print progress and a running time estimate. Write the
    script only, do not run it.

Fits here use the fixed per-cell C recorded by probe_sweep.py (its CV refits
on the full training split at the chosen C, so the axis-one point estimates
should reproduce the sweep numbers up to lbfgs determinism; both are stored
and a mismatch above 0.005 is flagged). Resample indices are drawn once per
(layer, set) for axis one and once per layer for axis two, so the three
models of a cell see identical draws and paired model differences can be
computed from the saved output.

Outputs (paths are flags, defaults alongside the sweep output):
    results/probe_bootstrap.json    "meta" (seeds, resample counts, sweep and
        scores paths) + "heldout" records (layer/set/model: point estimates,
        2.5/97.5 percentiles for argmax4 and AUC, the sweep's values) +
        "train" records (layer/model, main set: mean and sd over refits).
        Rewritten atomically after every layer (axis one) and every
        layer/model cell (axis two) as a checkpoint.
    results/probe_heldout_scores.npz    "scores_<set>_<model>" float32
        (n_layers, n_heldout_items, 4) held-out decision scores from the
        fixed-C fits, plus "heldout_ids_<set>", so later analysis does not
        need to refit.

Usage:
    /root/venv/bin/python scripts/probe_bootstrap.py \
        [--sweep results/probe_sweep.json] [--layer-dir /workspace/cand_end_layers] \
        [--out results/probe_bootstrap.json] [--scores-out results/probe_heldout_scores.npz] \
        [--heldout-resamples 1000] [--train-resamples 30] [--train-layers 0 15 31]
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from probe_one_layer import argmax_acc, build_examples

REPO_ROOT = Path(__file__).resolve().parent.parent
SET_FILES = {"main": REPO_ROOT / "data" / "main_set.json",
             "control": REPO_ROOT / "data" / "control_set.json"}
# Distinct from split seed 42 and shuffle seeds 1001/1002 (plan.md T2).
HELDOUT_BOOTSTRAP_SEED = 2001
TRAIN_BOOTSTRAP_SEED = 2002


def load_set(path: Path) -> dict:
    d = json.loads(path.read_text())
    train_ids, heldout_ids = d["train_indices"], d["heldout_indices"]
    assert not set(train_ids) & set(heldout_ids), f"{path}: split overlap"
    answers = {i: it["answer"] for i, it in enumerate(d["items"])}
    return {"train_ids": train_ids, "heldout_ids": heldout_ids,
            "answers": answers,
            "answers_ho": np.array([answers[i] for i in heldout_ids]),
            "n_items": len(d["items"])}


def fit_fixed_c(X_tr, y_tr, X_ho, c: float):
    """Standardize on X_tr (as in probe_one_layer.fit_probe) and fit logistic
    regression at fixed C; return held-out decision scores."""
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    clf = LogisticRegression(C=c, solver="lbfgs", max_iter=1000)
    clf.fit((X_tr - mu) / sd, y_tr)
    return clf.decision_function((X_ho - mu) / sd)


def bootstrap_metrics(scores, answers_ho, idx):
    """argmax4 and AUC per item-level resample. scores (n_ho*4,), idx
    (n_resamples, n_ho) item positions drawn with replacement."""
    per_item = scores.reshape(-1, 4)
    correct = per_item.argmax(axis=1) == answers_ho
    onehot = np.eye(4, dtype=bool)[answers_ho]
    argmax_draws = correct[idx].mean(axis=1)
    auc_draws = np.array([
        roc_auc_score(onehot[row].ravel(), per_item[row].ravel())
        for row in idx])
    return argmax_draws, auc_draws


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bootstrap error bars for the T5 probe curves")
    ap.add_argument("--sweep", type=Path,
                    default=REPO_ROOT / "results" / "probe_sweep.json",
                    help="sweep JSON holding the chosen C per layer/set/model")
    ap.add_argument("--layer-dir", type=Path,
                    default=Path("/workspace/cand_end_layers"))
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "results" / "probe_bootstrap.json")
    ap.add_argument("--scores-out", type=Path,
                    default=REPO_ROOT / "results" / "probe_heldout_scores.npz")
    ap.add_argument("--heldout-resamples", type=int, default=1000)
    ap.add_argument("--train-resamples", type=int, default=30)
    ap.add_argument("--train-layers", type=int, nargs="+", default=[0, 15, 31])
    args = ap.parse_args()

    sweep = json.loads(args.sweep.read_text())
    n_layers = sweep["meta"]["n_layers"]
    models = list(sweep["meta"]["models"])
    chosen_c = {(r["layer"], r["set"], r["model"]): r["chosen_C"]
                for r in sweep["records"]}
    sweep_vals = {(r["layer"], r["set"], r["model"]): r
                  for r in sweep["records"]}
    assert all((la, se, mo) in chosen_c for la in range(n_layers)
               for se in SET_FILES for mo in models), "sweep is incomplete"
    assert all(0 <= la < n_layers for la in args.train_layers), (
        f"--train-layers must be within 0..{n_layers - 1}")

    sets = {name: load_set(path) for name, path in SET_FILES.items()}

    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sweep_file": str(args.sweep),
        "layer_dir": str(args.layer_dir),
        "scores_file": str(args.scores_out),
        "heldout_bootstrap_seed": HELDOUT_BOOTSTRAP_SEED,
        "train_bootstrap_seed": TRAIN_BOOTSTRAP_SEED,
        "seed_note": "axis-one draws use default_rng([heldout_seed, layer, "
                     "set_index]); axis-two default_rng([train_seed, layer]); "
                     "identical draws across the three models of a cell",
        "n_heldout_resamples": args.heldout_resamples,
        "n_train_resamples": args.train_resamples,
        "train_layers": args.train_layers,
        "percentiles": [2.5, 97.5],
    }
    heldout_recs, train_recs = [], []

    def checkpoint():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(
            {"meta": meta, "heldout": heldout_recs, "train": train_recs},
            indent=1))
        tmp.replace(args.out)

    # ---- Axis one: held-out item bootstrap, all layers, both sets ----
    scores_store = {}  # (set, model) -> list of (n_ho, 4) per layer
    t0 = time.time()
    for layer in range(n_layers):
        payload = torch.load(args.layer_dir / f"layer_{layer:02d}.pt")
        assert payload["meta"]["layer"] == layer
        for iset, (set_name, s) in enumerate(sets.items()):
            assert payload["item_indices"][set_name] == list(range(s["n_items"]))
            n_ho = len(s["heldout_ids"])
            idx = np.random.default_rng(
                [HELDOUT_BOOTSTRAP_SEED, layer, iset]).integers(
                0, n_ho, size=(args.heldout_resamples, n_ho))
            for model in models:
                acts = payload["acts"][set_name][model]
                X_tr, y_tr, _ = build_examples(acts, s["train_ids"], s["answers"])
                X_ho, y_ho, _ = build_examples(acts, s["heldout_ids"], s["answers"])
                c = chosen_c[(layer, set_name, model)]
                sc = fit_fixed_c(X_tr, y_tr, X_ho, c)
                scores_store.setdefault((set_name, model), []).append(
                    sc.reshape(n_ho, 4).astype(np.float32))
                am, auc = argmax_acc(sc, s["answers_ho"]), float(
                    roc_auc_score(y_ho, sc))
                am_d, auc_d = bootstrap_metrics(sc, s["answers_ho"], idx)
                sw = sweep_vals[(layer, set_name, model)]
                rec = {
                    "layer": layer, "set": set_name, "model": model, "C": c,
                    "argmax4": am, "auc": auc,
                    "argmax4_lo": float(np.percentile(am_d, 2.5)),
                    "argmax4_hi": float(np.percentile(am_d, 97.5)),
                    "auc_lo": float(np.percentile(auc_d, 2.5)),
                    "auc_hi": float(np.percentile(auc_d, 97.5)),
                    "sweep_argmax4": sw["argmax4"], "sweep_auc": sw["auc"],
                }
                heldout_recs.append(rec)
                if abs(am - sw["argmax4"]) > 0.005 or abs(auc - sw["auc"]) > 0.005:
                    print(f"WARNING L{layer:02d} {set_name}/{model}: fixed-C "
                          f"refit differs from sweep (argmax4 {am:.4f} vs "
                          f"{sw['argmax4']:.4f}, AUC {auc:.4f} vs "
                          f"{sw['auc']:.4f})", flush=True)
        checkpoint()
        done, elapsed = layer + 1, time.time() - t0
        print(f"axis1 L{layer:02d} done ({done}/{n_layers}, "
              f"{elapsed / 60:.1f} min elapsed, "
              f"~{elapsed / done * (n_layers - done) / 60:.0f} min left)",
              flush=True)

    args.scores_out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"scores_{se}_{mo}": np.stack(v)
              for (se, mo), v in scores_store.items()}
    for set_name, s in sets.items():
        arrays[f"heldout_ids_{set_name}"] = np.array(s["heldout_ids"])
    np.savez(args.scores_out, **arrays)
    print(f"wrote held-out scores to {args.scores_out}", flush=True)

    # ---- Axis two: training-subsample bootstrap, main set only ----
    s = sets["main"]
    n_tr = len(s["train_ids"])
    t1 = time.time()
    n_cells = len(args.train_layers) * len(models)
    cell = 0
    for layer in args.train_layers:
        payload = torch.load(args.layer_dir / f"layer_{layer:02d}.pt")
        idx = np.random.default_rng([TRAIN_BOOTSTRAP_SEED, layer]).integers(
            0, n_tr, size=(args.train_resamples, n_tr))
        for model in models:
            acts = payload["acts"]["main"][model]
            X_ho, y_ho, _ = build_examples(acts, s["heldout_ids"], s["answers"])
            c = chosen_c[(layer, "main", model)]
            ams, aucs = [], []
            for row in idx:
                ids = [s["train_ids"][j] for j in row]
                X_tr, y_tr, _ = build_examples(acts, ids, s["answers"])
                sc = fit_fixed_c(X_tr, y_tr, X_ho, c)
                ams.append(argmax_acc(sc, s["answers_ho"]))
                aucs.append(float(roc_auc_score(y_ho, sc)))
            train_recs.append({
                "layer": layer, "set": "main", "model": model, "C": c,
                "n_resamples": args.train_resamples,
                "argmax4_mean": float(np.mean(ams)),
                "argmax4_sd": float(np.std(ams, ddof=1)),
                "auc_mean": float(np.mean(aucs)),
                "auc_sd": float(np.std(aucs, ddof=1)),
            })
            checkpoint()
            cell += 1
            elapsed = time.time() - t1
            print(f"axis2 L{layer:02d} {model}: argmax4 sd "
                  f"{train_recs[-1]['argmax4_sd']:.4f}, auc sd "
                  f"{train_recs[-1]['auc_sd']:.4f} ({cell}/{n_cells} cells, "
                  f"{elapsed / 60:.1f} min elapsed, "
                  f"~{elapsed / cell * (n_cells - cell) / 60:.0f} min left)",
                  flush=True)

    print(f"\nDone: {len(heldout_recs)} held-out records, {len(train_recs)} "
          f"train records in {args.out}; scores in {args.scores_out}")


if __name__ == "__main__":
    main()
