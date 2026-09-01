"""Plot per-position residual norm tables parsed from a caching log.

Implements the per-layer norm plot deliverable of docs/plan.md T4; the
intervention-layer marking convention is fixed in docs/plan.md T2.

Prompt (2026-08-31):
    Write scripts/plot_norms.py, which parses the per-position norm tables
    out of results/eval/cache_full_20260831.log and plots them. The log has
    one table per model and item set, each with columns layer, pos0, q_end,
    answer_marker, cand_end for 32 layers. Produce one figure per model-set
    combination with the four positions as lines against layer index, log
    scale on y since pos0 runs about ten times larger than the rest, and
    save as PNG under results/. Also produce a figure comparing the three
    models on pos0 alone, and one comparing them on cand_end alone. Take the
    log path and an output directory as arguments.

2026-08-31, appended prompt (intervention marks, main-set-only comparisons):
    Two changes. First, the intervention layers (5, 10, 15, 20, 25, 30) need
    vertical marks on every figure, per the convention in plan.md T2.
    Second, for the comparison figures, use main set only, three lines, so
    the model differences are easy to see; if the control set is worth
    showing, make it separate figures rather than dashed lines on the same
    axes.

Log format: each table is headed by
    === Per-position norm check, <model> / <set> (...) ===
followed by a column-header line and 32 rows: layer pos0 q_end answer_marker
cand_end. Table rows end at the first line that does not parse as one.

Outputs, written to the given output directory:
    norms_<model>_<set>.png     one per table: the four positions vs layer,
                                log-scale y
    compare_pos0_<set>.png      the three models on pos0, one set per figure,
                                log-scale y
    compare_cand_end_<set>.png  same for cand_end, linear y

Run as: .venv/bin/python scripts/plot_norms.py results/eval/cache_full_20260831.log results/figures/norms/
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLUMNS = ["pos0", "q_end", "answer_marker", "cand_end"]
INTERVENTION_LAYERS = [5, 10, 15, 20, 25, 30]  # docs/plan.md T2
HEADER_RE = re.compile(r"^=== Per-position norm check, (\S+) / (\S+) ")
ROW_RE = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def parse_log(log_path: Path) -> dict[tuple[str, str], dict[str, list[float]]]:
    """Return {(model, set): {column: [value per layer]}}, layers in row order."""
    tables: dict[tuple[str, str], dict[str, list[float]]] = {}
    current = None
    for line in log_path.read_text().splitlines():
        header = HEADER_RE.match(line)
        if header:
            current = {col: [] for col in COLUMNS}
            current["layer"] = []
            tables[(header.group(1), header.group(2))] = current
            continue
        if current is None:
            continue
        row = ROW_RE.match(line)
        if row:
            current["layer"].append(int(row.group(1)))
            for col, val in zip(COLUMNS, row.groups()[1:]):
                current[col].append(float(val))
        elif current["layer"]:
            current = None  # table ended (progress line etc.)
    return tables


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


def plot_per_table(tables, out_dir: Path) -> None:
    for (model, item_set), tab in tables.items():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for col in COLUMNS:
            ax.plot(tab["layer"], tab[col], marker=".", label=col)
        mark_intervention_layers(ax)
        ax.set_yscale("log")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean L2 norm (log scale)")
        ax.set_title(f"Residual norms by position: {model} / {item_set}")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        out = out_dir / f"norms_{model}_{item_set}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


def plot_comparison(tables, column: str, item_set: str, log_y: bool, out_dir: Path) -> None:
    """One figure: the three models on one column, one item set."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (model, s), tab in sorted(tables.items()):
        if s != item_set:
            continue
        ax.plot(tab["layer"], tab[column], marker=".", label=model)
    mark_intervention_layers(ax)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Mean L2 norm at {column}" + (" (log scale)" if log_y else ""))
    ax.set_title(f"Model comparison: {column} ({item_set} set)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    out = out_dir / f"compare_{column}_{item_set}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log_path", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()

    tables = parse_log(args.log_path)
    assert tables, f"no norm tables found in {args.log_path}"
    for (model, item_set), tab in tables.items():
        n = len(tab["layer"])
        assert n == 32, f"{model}/{item_set}: expected 32 layers, got {n}"
        print(f"parsed {model}/{item_set}: {n} layers")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_per_table(tables, args.out_dir)
    for item_set in sorted({s for _, s in tables}):
        plot_comparison(tables, "pos0", item_set, log_y=True, out_dir=args.out_dir)
        plot_comparison(tables, "cand_end", item_set, log_y=False, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
