"""Plot the T5 probe sweep curves from results/probe_sweep.json.

Implements the headline-figure deliverable of docs/plan.md T5; metrics,
baselines, and the intervention-layer marking convention are fixed in
docs/plan.md T2.

Prompt (2026-09-01):
    Write scripts/plot_probe_sweep.py, reading results/probe_sweep.json and
    producing the T5 figures per the conventions in plan.md T2. Four figures:
    argmax4 against layer for the main set, argmax4 for the control set, AUC
    for main, AUC for control. Each has three lines, one per model, with a
    legend. Mark the intervention layers (5, 10, 15, 20, 25, 30) with
    vertical lines as in plot_norms.py. Draw the chance line at 0.25 on the
    argmax figures and 0.5 on the AUC figures. Also plot the corresponding
    shuffled baseline as a separate line per model, in a lighter style, so
    the reader can see it sits at chance. Save PNGs under results/. Take the
    JSON path and an output directory as arguments. Write the script only,
    do not run it.

Input: the JSON written by scripts/probe_sweep.py — a "meta" block and a
flat "records" list with one record per (layer, set, model) carrying
argmax4, auc, acc, chosen_C, shuf_argmax4 (within-item shuffle, read on
argmax4), and shuf_auc (global shuffle, read on AUC).

Outputs, written to the given output directory:
    probe_argmax4_main.png      within-item argmax accuracy vs layer, main set
    probe_argmax4_control.png   same, control set
    probe_auc_main.png          held-out AUC vs layer, main set
    probe_auc_control.png       same, control set
Each figure: one solid line per model, the matching shuffled baseline per
model as a lighter dashed line in the same color, the analytic chance level
as a horizontal dashed line, and the intervention layers as dotted verticals.

Run as: .venv/bin/python scripts/plot_probe_sweep.py results/probe_sweep.json results/
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INTERVENTION_LAYERS = [5, 10, 15, 20, 25, 30]  # docs/plan.md T2
SETS = ["main", "control"]
# metric key -> (shuffled-baseline key, chance level, axis label)
METRICS = {
    "argmax4": ("shuf_argmax4", 0.25, "Within-item argmax accuracy"),
    "auc": ("shuf_auc", 0.5, "Held-out AUC"),
}


def mark_intervention_layers(ax) -> None:
    for i, layer in enumerate(INTERVENTION_LAYERS):
        ax.axvline(
            layer,
            color="gray",
            linestyle=":",
            linewidth=1,
            alpha=0.7,
            label="intervention layers" if i == 0 else None,
        )


def collect(records, item_set: str, model: str, key: str,
            n_layers: int) -> list[float]:
    """Values of one record field for (item_set, model), ordered by layer."""
    by_layer = {r["layer"]: r[key] for r in records
                if r["set"] == item_set and r["model"] == model}
    assert sorted(by_layer) == list(range(n_layers)), (
        f"{item_set}/{model}: layers present {sorted(by_layer)} do not cover "
        f"0..{n_layers - 1}; is the sweep complete?"
    )
    return [by_layer[layer] for layer in range(n_layers)]


def plot_metric(records, models: list[str], item_set: str, metric: str,
                n_layers: int, out_dir: Path) -> None:
    shuf_key, chance, ylabel = METRICS[metric]
    layers = list(range(n_layers))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, model in enumerate(models):
        color = f"C{i}"
        ax.plot(layers, collect(records, item_set, model, metric, n_layers),
                marker=".", color=color, label=model)
        ax.plot(layers, collect(records, item_set, model, shuf_key, n_layers),
                linestyle="--", linewidth=1, alpha=0.35, color=color,
                label="shuffled baselines (per model)" if i == 0 else None)
    ax.axhline(chance, color="black", linestyle="--", linewidth=1, alpha=0.5,
               label=f"chance {chance}")
    mark_intervention_layers(ax)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Probe {ylabel.lower()} at cand_end ({item_set} set)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / f"probe_{metric}_{item_set}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_path", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()

    data = json.loads(args.json_path.read_text())
    records, meta = data["records"], data["meta"]
    n_layers = meta["n_layers"]
    models = list(meta["models"])
    assert len(models) == 3, f"expected 3 models, meta lists {models}"
    print(f"loaded {len(records)} records: {n_layers} layers, sets {SETS}, "
          f"models {models}")
    print(f"shuffle seeds: within-item {meta['within_item_shuffle_seed']} "
          f"(argmax4), global {meta['global_shuffle_seed']} (AUC)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for item_set in SETS:
        for metric in METRICS:
            plot_metric(records, models, item_set, metric, n_layers,
                        args.out_dir)


if __name__ == "__main__":
    main()
