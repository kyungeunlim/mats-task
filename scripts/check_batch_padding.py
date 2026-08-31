"""Check that right-padding in cache_activations.py batching is inert (plan.md T4).

Prompt (2026-08-31, sanity check for the batching in scripts/cache_activations.py):
    Write scripts/check_batch_padding.py, which tests whether the
    right-padding in cache_activations.py's batching affects the cached
    activations. It takes two directories of cached chunks produced by the
    same items and model but different --batch-items settings, loads the
    matching chunk from each, and reports: per-item max abs diff alongside
    per-item max|a|; a per-layer table of max abs diff, max|a|, and their
    ratio; and per-position max abs diff labelled q_end, answer_marker,
    cand_end. Take the two directories as arguments. Print what the reading
    means: a flat ratio across layers indicates rounding rather than
    contamination, since contamination would enter where attention first
    mixes padded tokens and grow as a fraction of the residual. Also print
    the bf16 ULP equivalent of the worst ratio. Include in the docstring the
    reason padding should be inert (causal attention, rotary positions, all
    three cached positions at or before the real sequence end) and note that
    this measurement is consistent with rounding rather than proof of no
    contamination, since a sub-ULP padding bug would not be caught.

2026-08-31, appended prompt (explicit path shape):
    In scripts/check_batch_padding.py, make the expected path shape
    explicit. The two arguments are the leaf directories containing
    chunk_*.pt, not the --out-dir root that cache_activations.py takes, and
    the error message when no chunks are found should say so, for example
    "no chunk_*.pt files in X (expected a leaf directory like
    OUT_DIR/MODEL/SET)". Update the argument help text the same way, and
    add a usage example to the docstring.

Why padding should be inert: attention in these models is causal and the
position embedding is rotary, so the token at index i attends only to indices
<= i and its rotary phase depends only on its own index, which right padding
does not shift. Pad tokens sit strictly after the real tokens, and all three
cached positions (q_end, answer_marker, cand_end) are at or before the real
sequence end, so no cached activation has a pad token in its causal past.
Different --batch-items settings do change the padded batch SHAPE, which
changes kernel choice and floating-point summation order, so bf16-rounding-
level differences between the two runs are expected and benign.

Limitation: this measurement is consistent with rounding rather than proof of
no contamination. A padding bug whose effect at the cached positions stays
below one bf16 ULP would be indistinguishable from rounding here and would
not be caught.

Usage:
    /root/venv/bin/python scripts/check_batch_padding.py \
        /workspace/activations_b1/unfiltered/main \
        /workspace/activations_b4/unfiltered/main
Each argument is the LEAF directory that directly contains the chunk_*.pt
files, shaped OUT_DIR/MODEL/SET — not the --out-dir root that
cache_activations.py takes. Chunks present in both directories
(matched by filename, so run both with the same --limit/--chunk-items) are
compared; positions, item indices, model, and revision are asserted equal.
"""

import argparse
import sys
from pathlib import Path

import torch

BF16_ULP = 2.0 ** -8  # relative spacing of bfloat16 (8-bit effective mantissa)
POSITION_NAMES = ("q_end", "answer_marker", "cand_end")
MAX_ITEM_ROWS = 64  # per-item table cap; above this, summarize + worst rows


def load_chunks(dir_path: Path) -> dict[str, Path]:
    chunks = {p.name: p for p in sorted(dir_path.glob("chunk_*.pt"))}
    if not chunks:
        sys.exit(f"no chunk_*.pt files in {dir_path} (expected a leaf "
                 f"directory like OUT_DIR/MODEL/SET, e.g. "
                 f"/workspace/activations/unfiltered/main, not the "
                 f"--out-dir root that cache_activations.py takes)")
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare cached activations from two --batch-items runs"
    )
    parser.add_argument(
        "dir_a", type=Path,
        help="leaf directory directly containing chunk_*.pt files "
             "(OUT_DIR/MODEL/SET, not the --out-dir root that "
             "cache_activations.py takes)")
    parser.add_argument(
        "dir_b", type=Path,
        help="second leaf directory, same shape as dir_a")
    args = parser.parse_args()

    chunks_a, chunks_b = load_chunks(args.dir_a), load_chunks(args.dir_b)
    common = sorted(set(chunks_a) & set(chunks_b))
    if not common:
        sys.exit(f"no chunk filenames in common between {args.dir_a} and "
                 f"{args.dir_b}; run both caches with the same --limit and "
                 f"--chunk-items")
    only_a = sorted(set(chunks_a) - set(chunks_b))
    only_b = sorted(set(chunks_b) - set(chunks_a))
    if only_a or only_b:
        print(f"note: unmatched chunks ignored (A only: {only_a}, "
              f"B only: {only_b})")

    diffs, mags, item_ids = [], [], []
    meta_a = meta_b = None
    for name in common:
        a = torch.load(chunks_a[name])
        b = torch.load(chunks_b[name])
        meta_a, meta_b = a["meta"], b["meta"]
        for key in ("model", "revision", "dtype"):
            assert meta_a[key] == meta_b[key], (
                f"{name}: meta[{key!r}] differs: {meta_a[key]} vs {meta_b[key]}"
            )
        assert a["item_indices"] == b["item_indices"], f"{name}: item indices differ"
        assert torch.equal(a["positions"], b["positions"]), (
            f"{name}: cached position indices differ; the runs disagree on "
            f"tokenization, not just batching"
        )
        assert a["acts"].shape == b["acts"].shape, f"{name}: shape mismatch"
        diffs.append((a["acts"].float() - b["acts"].float()).abs())
        mags.append(a["acts"].float().abs())
        item_ids.extend(a["item_indices"])

    diff = torch.cat(diffs)   # (items, cand, layer, pos, d_model)
    mag = torch.cat(mags)
    n_items, _, n_layers, n_pos, _ = diff.shape
    print(f"\nCompared {len(common)} chunk(s), {n_items} items, "
          f"model {meta_a['model']} rev {meta_a['revision'][:12]}, "
          f"dtype {meta_a['dtype']}")
    print(f"A: {args.dir_a}\nB: {args.dir_b}")

    overall = diff.max().item()
    if overall == 0.0:
        print("\nWARNING: activations are bitwise identical. That is not "
              "expected across different --batch-items settings (batch shape "
              "changes summation order); check that the two directories "
              "really came from different settings.")

    # ---- per item -----------------------------------------------------------
    item_diff = diff.amax(dim=(1, 2, 3, 4))
    item_mag = mag.amax(dim=(1, 2, 3, 4))
    print(f"\n=== Per-item max abs diff (alongside per-item max|a|) ===")
    rows = range(n_items)
    if n_items > MAX_ITEM_ROWS:
        worst = torch.topk(item_diff, 8).indices.tolist()
        print(f"({n_items} items; showing the 8 worst, "
              f"full stats: mean {item_diff.mean():.3e}, max {overall:.3e})")
        rows = sorted(worst)
    print(f"{'item':>6} {'max|A-B|':>12} {'max|a|':>10}")
    for i in rows:
        print(f"{item_ids[i]:>6} {item_diff[i].item():>12.4e} "
              f"{item_mag[i].item():>10.3f}")

    # ---- per layer ----------------------------------------------------------
    layer_diff = diff.amax(dim=(0, 1, 3, 4))
    layer_mag = mag.amax(dim=(0, 1, 3, 4))
    ratio = layer_diff / layer_mag
    print(f"\n=== Per-layer table ===")
    print(f"{'layer':>5} {'max|A-B|':>12} {'max|a|':>10} {'ratio':>10}")
    for l in range(n_layers):
        print(f"{l:>5} {layer_diff[l].item():>12.4e} "
              f"{layer_mag[l].item():>10.3f} {ratio[l].item():>10.3e}")

    # ---- per position -------------------------------------------------------
    pos_diff = diff.amax(dim=(0, 1, 2, 4))
    print(f"\n=== Per-position max abs diff ===")
    for p in range(n_pos):
        print(f"{POSITION_NAMES[p]:>14}: {pos_diff[p].item():.4e}")

    # ---- reading ------------------------------------------------------------
    worst_ratio = ratio.max().item()
    print(f"\n=== Reading ===")
    print("Contamination from padded tokens would enter at the layer where "
          "attention first mixes them into a cached position and then grow "
          "as a fraction of the residual, so the ratio column would rise "
          "with depth from that entry layer. A flat ratio across layers "
          "indicates rounding: different batch shapes change kernel choice "
          "and summation order, nothing else.")
    print(f"Worst per-layer ratio: {worst_ratio:.3e} = "
          f"{worst_ratio / BF16_ULP:.2f} bf16 ULPs of that layer's max |a|.")
    print("Caveat: this is consistent with rounding, not proof of no "
          "contamination; a sub-ULP padding bug would not be caught.")


if __name__ == "__main__":
    main()
