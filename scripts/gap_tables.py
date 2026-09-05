"""Per-layer model gaps behind the write-up's main finding, with a paired
bootstrap interval on each gap and a two-panel figure (T5/T9 support).

Prints, for each layer and each metric, base minus CB and base minus
filtered on the main and control sets, read from results/probe_sweep.json,
with a summary over layers 15 to 31 and the per-point bootstrap
uncertainty from results/probe_bootstrap.json for a few deep layers. Then
recomputes the held-out item bootstrap from results/probe_heldout_scores.npz
as a paired bootstrap on the gaps, and plots the gaps with those intervals.

Prompt (2026-09-02):
    Write scripts/gap_tables.py, which reads results/probe_sweep.json and
    results/probe_bootstrap.json and prints the per-layer model gaps that
    the write-up's main finding rests on. For each layer 0 to 31 and each
    metric (argmax4 and AUC), print four columns: base minus the fine-tune
    on main, base minus the fine-tune on control, base minus filtered on
    main, base minus filtered on control. Alongside each table, print a
    summary over layers 15 to 31: how many of the 17 layers are positive
    for each column, and the mean gap. Also print, for a few deep layers,
    the item-axis bootstrap half-width and the fit-axis standard deviation
    from probe_bootstrap.json, so the gaps can be read against the
    per-point uncertainty. Take the two JSON paths as optional arguments
    with the current defaults. Write the script only, do not run it.

2026-09-02, appended prompt (paired bootstrap and figure):
    Also add a paired bootstrap on the gaps. The item-axis draws are
    identical across the three models of a cell, so resampling held-out
    items from results/probe_heldout_scores.npz with the same seed and
    recomputing each model's metric on the same draw gives a paired
    difference per draw. Report the 2.5 and 97.5 percentiles of the
    base-minus-model difference per layer, per set, per metric. This is
    the correct interval for the gap; the per-model half-widths currently
    printed are conservative because the item variation is shared and
    cancels in the paired comparison.

    Then produce a figure: two panels, one for base minus filtered and one
    for base minus the fine-tune, each showing the main and control gaps
    against layer, with a horizontal line at zero and the paired bootstrap
    interval shaded per line. Mark the intervention layers as in
    plot_norms.py. Save as PNG under an output directory given as an
    argument.

2026-09-04, appended prompt (relative paths in output):
    Print and record paths relative to the repo root instead of absolute,
    so logs and result files do not carry the laptop's home directory.
    Output-only change; no computation touched.

Model keys follow probe_sweep.json "meta.models": base = unfiltered,
fine-tune = unlearned-cb (circuit breakers), filtered = e2e-strong-filter.

Point-estimate gaps in the tables come from probe_sweep.json (the CV
refit). The paired intervals come from the npz scores, which are the
fixed-C refit of probe_bootstrap.py; the two fits agree to within 0.005
on main and differ in a few control cells (docs/results.md, T5). The
figure's lines are the npz point gaps, so line and band share one fit; the
sweep gap is printed alongside in the paired table for comparison.

Paired bootstrap. The draws reproduce probe_bootstrap.py exactly:
default_rng([heldout_bootstrap_seed, layer, iset]).integers(0, n_ho,
size=(n_resamples, n_ho)) with iset 0 = main, 1 = control, seed and
resample count read from probe_bootstrap.json "meta". For each draw the
metric is computed for all three models on the same resampled items, and
the base-minus-model difference is taken per draw; the 2.5/97.5
percentiles of those differences are the paired interval. As a check that
the draws were reproduced, the per-model percentiles recomputed here are
compared with the lo/hi stored in probe_bootstrap.json and the largest
absolute discrepancy is printed (expected 0 up to float rounding).

AUC is computed with a numpy Mann-Whitney rank statistic using average
ranks for ties, which equals sklearn.metrics.roc_auc_score; sklearn is not
imported so the script runs on the laptop venv.

Outputs:
    stdout tables (per-layer gaps, deep-layer summary, per-point
        uncertainty, paired-bootstrap intervals)
    OUT_DIR/gap_argmax4.png, OUT_DIR/gap_auc.png    two panels each
        (base minus filtered, base minus CB), main and control lines with
        paired 95% bootstrap bands, zero line, intervention layers marked.
    OUT_DIR/gap_paired_bootstrap.json    the paired intervals, so the
        write-up can quote them without rerunning.

Usage:
    python scripts/gap_tables.py OUT_DIR [--sweep results/probe_sweep.json]
        [--bootstrap results/probe_bootstrap.json]
        [--scores results/probe_heldout_scores.npz]
        [--deep-layers 15 20 25 31] [--summary-from 15]
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SET_FILES = {"main": REPO_ROOT / "data" / "main_set.json",
             "control": REPO_ROOT / "data" / "control_set.json"}
SETS = ["main", "control"]  # iset order in probe_bootstrap.py
BASE, FINETUNE, FILTERED = "unfiltered", "unlearned-cb", "e2e-strong-filter"
SHORT = {FINETUNE: "CB", FILTERED: "filt"}
LONG = {FINETUNE: "fine-tune (CB)", FILTERED: "filtered"}
METRICS = ("argmax4", "auc")
METRIC_LABEL = {"argmax4": "argmax4 accuracy", "auc": "AUC"}
COLUMNS = [  # (label, set, subtracted model)
    ("base-CB main", "main", FINETUNE),
    ("base-CB ctrl", "control", FINETUNE),
    ("base-filt main", "main", FILTERED),
    ("base-filt ctrl", "control", FILTERED),
]
INTERVENTION_LAYERS = [5, 10, 15, 20, 25, 30]  # docs/plan.md T2, as plot_norms.py
# Fixed per-entity colors (main, control), same blue/orange pair as the
# other probe figures; matplotlib C0/C1 pass the CVD separation check.
SET_STYLE = {"main": dict(color="C0", linestyle="-"),
             "control": dict(color="C1", linestyle="--")}
BAND_CAPTION = ("Bands: paired 95% bootstrap interval over held-out items "
                "(both models scored on the same resampled items, difference "
                "per draw). Lines: fixed-C refit gaps from the saved scores. "
                "Fit variance is not included.")


def rel(p: Path) -> str:
    """Path relative to the repo root when inside it, else as given. Used only
    for printing and for recording paths in output files, so logs and result
    JSONs do not carry the machine's home directory."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def index_records(records: list[dict]) -> dict:
    return {(r["layer"], r["set"], r["model"]): r for r in records}


def mark_intervention_layers(ax) -> None:
    for i, layer in enumerate(INTERVENTION_LAYERS):
        ax.axvline(layer, color="gray", linestyle=":", linewidth=1, alpha=0.7,
                   label="intervention layers" if i == 0 else None)


def average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based average ranks with ties sharing their mean rank."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1  # mean of 1-based ranks i+1..j+1
        i = j + 1
    return ranks


def auc_rank(scores: np.ndarray, pos: np.ndarray) -> float:
    """Mann-Whitney AUC = P(score_pos > score_neg) + 0.5 P(tie); equals
    sklearn roc_auc_score for binary labels."""
    r = average_ranks(scores)
    n_pos = int(pos.sum())
    n_neg = len(scores) - n_pos
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def draw_metrics(per_item: np.ndarray, answers: np.ndarray, idx: np.ndarray):
    """argmax4 and AUC for each item-level resample. per_item (n_ho, 4),
    answers (n_ho,), idx (n_resamples, n_ho). Same computation as
    probe_bootstrap.bootstrap_metrics."""
    correct = per_item.argmax(axis=1) == answers
    onehot = np.eye(4, dtype=bool)[answers]
    argmax_draws = correct[idx].mean(axis=1)
    auc_draws = np.array([auc_rank(per_item[row].ravel(), onehot[row].ravel())
                          for row in idx])
    return argmax_draws, auc_draws


def paired_bootstrap(z, sets_answers: dict, seed: int, n_resamples: int,
                     n_layers: int, ho: dict):
    """Return (paired, point, max_check) where
    paired[(metric, set, other)] = (n_layers, 2) lo/hi of base minus other,
    point[(metric, set, model)] = (n_layers,) fixed-C point estimates,
    max_check = largest |recomputed per-model percentile - stored| ."""
    paired, point, max_check = {}, {}, 0.0
    models = (BASE, FINETUNE, FILTERED)
    for iset, set_name in enumerate(SETS):
        answers = sets_answers[set_name]
        n_ho = len(answers)
        scores = {m: z[f"scores_{set_name}_{m}"] for m in models}
        for m in models:
            assert scores[m].shape == (n_layers, n_ho, 4), scores[m].shape
        for layer in range(n_layers):
            idx = np.random.default_rng([seed, layer, iset]).integers(
                0, n_ho, size=(n_resamples, n_ho))
            draws = {}
            for m in models:
                am_d, auc_d = draw_metrics(scores[m][layer], answers, idx)
                draws[m] = {"argmax4": am_d, "auc": auc_d}
                per_item = scores[m][layer]
                point.setdefault(("argmax4", set_name, m), np.zeros(n_layers))[layer] = \
                    float((per_item.argmax(axis=1) == answers).mean())
                point.setdefault(("auc", set_name, m), np.zeros(n_layers))[layer] = \
                    auc_rank(per_item.ravel(), np.eye(4, dtype=bool)[answers].ravel())
                stored = ho.get((layer, set_name, m))
                if stored is not None:
                    for metric in METRICS:
                        lo, hi = np.percentile(draws[m][metric], [2.5, 97.5])
                        max_check = max(max_check,
                                        abs(lo - stored[f"{metric}_lo"]),
                                        abs(hi - stored[f"{metric}_hi"]))
            for other in (FINETUNE, FILTERED):
                for metric in METRICS:
                    diff = draws[BASE][metric] - draws[other][metric]
                    lo, hi = np.percentile(diff, [2.5, 97.5])
                    paired.setdefault((metric, set_name, other),
                                      np.zeros((n_layers, 2)))[layer] = (lo, hi)
    return paired, point, max_check


def plot_gaps(metric: str, paired: dict, point: dict, n_layers: int,
              n_resamples: int, out_dir: Path) -> Path:
    layers = np.arange(n_layers)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, other in zip(axes, (FILTERED, FINETUNE)):
        for i, set_name in enumerate(SETS):
            gap = point[(metric, set_name, BASE)] - point[(metric, set_name, other)]
            band = paired[(metric, set_name, other)]
            st = SET_STYLE[set_name]
            ax.fill_between(layers, band[:, 0], band[:, 1], color=st["color"],
                            alpha=0.15, linewidth=0,
                            label="paired 95% CI" if i == 0 else None)
            ax.plot(layers, gap, marker=".", linewidth=1.5, color=st["color"],
                    linestyle=st["linestyle"], label=f"{set_name} set")
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        mark_intervention_layers(ax)
        ax.set_title(f"base minus {LONG[other]}")
        ax.set_xlabel("layer")
        ax.set_xticks(range(0, n_layers, 5))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(f"gap in {METRIC_LABEL[metric]} (base minus model)")
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle(f"Probe gap vs layer at cand_end, {METRIC_LABEL[metric]}",
                 fontsize=11)
    fig.text(0.01, 0.01, BAND_CAPTION, fontsize=7, color="gray", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = out_dir / f"gap_{metric}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-layer base-minus-model probe gaps with paired "
                    "bootstrap intervals and a two-panel figure")
    ap.add_argument("out_dir", type=Path, help="directory for PNGs and JSON")
    ap.add_argument("--sweep", type=Path,
                    default=REPO_ROOT / "results" / "probe_sweep.json")
    ap.add_argument("--bootstrap", type=Path,
                    default=REPO_ROOT / "results" / "probe_bootstrap.json")
    ap.add_argument("--scores", type=Path,
                    default=REPO_ROOT / "results" / "probe_heldout_scores.npz")
    ap.add_argument("--summary-from", type=int, default=15,
                    help="first layer of the deep-layer summary (default 15)")
    ap.add_argument("--deep-layers", type=int, nargs="+",
                    default=[15, 20, 25, 31],
                    help="layers for the uncertainty table")
    args = ap.parse_args()

    sweep = json.loads(args.sweep.read_text())
    n_layers = sweep["meta"]["n_layers"]
    models = sweep["meta"]["models"]
    for m in (BASE, FINETUNE, FILTERED):
        assert m in models, f"{m} not in sweep meta.models: {list(models)}"
    sw = index_records(sweep["records"])
    boot = json.loads(args.bootstrap.read_text())
    ho = index_records(boot["heldout"])
    tr = index_records(boot["train"])
    seed = boot["meta"]["heldout_bootstrap_seed"]
    n_resamples = boot["meta"]["n_heldout_resamples"]

    # Held-out answers per set, aligned to the npz order.
    z = np.load(args.scores)
    sets_answers = {}
    for set_name in SETS:
        d = json.loads(SET_FILES[set_name].read_text())
        ids = z[f"heldout_ids_{set_name}"]
        assert list(ids) == list(d["heldout_indices"]), \
            f"{set_name}: npz held-out ids differ from set file"
        sets_answers[set_name] = np.array([d["items"][i]["answer"] for i in ids])

    print(f"sweep     : {rel(args.sweep)}")
    print(f"bootstrap : {rel(args.bootstrap)}")
    print(f"scores    : {rel(args.scores)}")
    print(f"base = {BASE} ({models[BASE]})")
    print(f"CB   = {FINETUNE} ({models[FINETUNE]})")
    print(f"filt = {FILTERED} ({models[FILTERED]})")
    print("gap = base value minus the named model's value; positive = base "
          "higher")
    summ_layers = list(range(args.summary_from, n_layers))

    # ---- paired bootstrap from the npz ----
    print(f"\nrecomputing held-out item bootstrap from npz: seed {seed}, "
          f"{n_resamples} resamples, draws default_rng([seed, layer, iset]) "
          f"with iset main=0, control=1 ...", flush=True)
    paired, point, max_check = paired_bootstrap(
        z, sets_answers, seed, n_resamples, n_layers, ho)
    print(f"check: max |recomputed per-model percentile - stored in "
          f"probe_bootstrap.json| = {max_check:.2e}"
          + ("" if max_check < 1e-6 else
             "   WARNING: draws or metrics do not reproduce the stored bands"))

    for metric in METRICS:
        # gaps[layer, column] from sweep point estimates
        gaps = np.full((n_layers, len(COLUMNS)), np.nan)
        for layer in range(n_layers):
            for j, (_, set_name, other) in enumerate(COLUMNS):
                b = sw.get((layer, set_name, BASE))
                o = sw.get((layer, set_name, other))
                if b is not None and o is not None:
                    gaps[layer, j] = b[metric] - o[metric]

        print("\n" + "=" * 72)
        print(f"{metric}: base minus model, per layer (sweep point estimates)")
        print("=" * 72)
        print(f"{'layer':>5} " + " ".join(f"{lab:>15}" for lab, _, _ in COLUMNS))
        for layer in range(n_layers):
            cells = " ".join(
                f"{'n/a':>15}" if np.isnan(v) else f"{v:>+15.4f}"
                for v in gaps[layer])
            print(f"{layer:>5} {cells}")

        print("-" * 72)
        print(f"summary over layers {summ_layers[0]}..{summ_layers[-1]} "
              f"({len(summ_layers)} layers)")
        sub = gaps[summ_layers]
        n_pos = (sub > 0).sum(axis=0)
        n_ok = (~np.isnan(sub)).sum(axis=0)
        # paired interval excludes zero (lo > 0) at the summary layers
        n_excl = []
        for _, set_name, other in COLUMNS:
            band = paired[(metric, set_name, other)][summ_layers]
            n_excl.append(int((band[:, 0] > 0).sum()))
        print(f"{'':>5} " + " ".join(f"{lab:>15}" for lab, _, _ in COLUMNS))
        print(f"{'n>0':>5} " + " ".join(
            f"{f'{p}/{k}':>15}" for p, k in zip(n_pos, n_ok)))
        print(f"{'CI>0':>5} " + " ".join(
            f"{f'{e}/{len(summ_layers)}':>15}" for e in n_excl)
              + "   (paired 2.5th percentile above zero)")
        print(f"{'mean':>5} " + " ".join(
            f"{m:>+15.4f}" for m in np.nanmean(sub, axis=0)))
        print(f"{'min':>5} " + " ".join(
            f"{m:>+15.4f}" for m in np.nanmin(sub, axis=0)))
        print(f"{'max':>5} " + " ".join(
            f"{m:>+15.4f}" for m in np.nanmax(sub, axis=0)))

        # ---- paired bootstrap intervals per layer ----
        print("-" * 72)
        print(f"{metric}: paired 95% bootstrap interval on the gap "
              f"(base minus model), from npz fixed-C scores")
        print(f"  gap = npz point gap; [lo, hi] = 2.5/97.5 percentiles of the "
              f"per-draw difference; sweep = gap from sweep point estimates")
        print(f"{'layer':>5} " + " ".join(
            f"{lab:>29}" for lab, _, _ in COLUMNS))
        print(f"{'':>5} " + " ".join(
            f"{'gap [lo, hi] sweep':>29}" for _ in COLUMNS))
        for layer in range(n_layers):
            cells = []
            for j, (_, set_name, other) in enumerate(COLUMNS):
                g = point[(metric, set_name, BASE)][layer] - \
                    point[(metric, set_name, other)][layer]
                lo, hi = paired[(metric, set_name, other)][layer]
                s = f"{g:+.3f} [{lo:+.3f},{hi:+.3f}]"
                s += "  n/a" if np.isnan(gaps[layer, j]) else f" {gaps[layer, j]:+.3f}"
                cells.append(f"{s:>29}")
            print(f"{layer:>5} " + " ".join(cells))

        # ---- per-point uncertainty for a few deep layers ----
        print("-" * 72)
        print(f"{metric}: per-point uncertainty at deep layers "
              f"(one model's point, not the paired gap; conservative for "
              f"gaps since item variation is shared)")
        print("  item hw = (97.5th - 2.5th percentile)/2 over "
              f"{n_resamples} held-out item resamples")
        print("  fit sd  = sd over "
              f"{boot['meta']['n_train_resamples']} training-item resamples "
              "at fixed C, main set only")
        print(f"{'layer':>5} {'model':>18} {'set':>8} {'point':>8} "
              f"{'item hw':>8} {'fit sd':>8}")
        for layer in args.deep_layers:
            for model in (BASE, FINETUNE, FILTERED):
                for set_name in SETS:
                    h = ho.get((layer, set_name, model))
                    t = tr.get((layer, set_name, model))
                    if h is None:
                        print(f"{layer:>5} {model:>18} {set_name:>8} "
                              f"{'n/a':>8}")
                        continue
                    hw = (h[f"{metric}_hi"] - h[f"{metric}_lo"]) / 2
                    sd = "n/a" if t is None else f"{t[f'{metric}_sd']:.4f}"
                    print(f"{layer:>5} {model:>18} {set_name:>8} "
                          f"{h[metric]:>8.4f} {hw:>8.4f} {sd:>8}")

    # ---- figures and JSON ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        out = plot_gaps(metric, paired, point, n_layers, n_resamples,
                        args.out_dir)
        print(f"\nwrote {rel(out)}")
    records = []
    for (metric, set_name, other), band in paired.items():
        for layer in range(n_layers):
            records.append({
                "layer": layer, "set": set_name, "metric": metric,
                "other": other,
                "gap": float(point[(metric, set_name, BASE)][layer]
                             - point[(metric, set_name, other)][layer]),
                "lo": float(band[layer, 0]), "hi": float(band[layer, 1]),
                "sweep_gap": float(sw[(layer, set_name, BASE)][metric]
                                   - sw[(layer, set_name, other)][metric]),
            })
    out_json = args.out_dir / "gap_paired_bootstrap.json"
    out_json.write_text(json.dumps({
        "meta": {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sweep_file": rel(args.sweep), "bootstrap_file": rel(args.bootstrap),
            "scores_file": rel(args.scores), "base": BASE,
            "heldout_bootstrap_seed": seed, "n_heldout_resamples": n_resamples,
            "draw_note": "default_rng([seed, layer, iset]).integers(0, n_ho, "
                         "size=(n_resamples, n_ho)); iset main=0, control=1; "
                         "same draws as probe_bootstrap.py",
            "interval": "2.5/97.5 percentiles of per-draw (base - other)",
            "per_model_percentile_check_max_abs_diff": max_check,
        },
        "records": records}, indent=1))
    print(f"wrote {rel(out_json)}")


if __name__ == "__main__":
    main()
