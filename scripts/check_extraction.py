"""Verify the per-layer cand_end extraction against source chunks (plan.md T5 prep).

Prompt (2026-08-31, checks the output of scripts/extract_cand_end.py):
    Write scripts/check_extraction.py, which verifies that the per-layer
    files from extract_cand_end.py faithfully reproduce the cand_end slice
    of the source chunks. Take the activations root and the extraction
    directory as arguments. For a sample of layers (all 32 by default, with
    a flag to sample fewer), and for every model and set combination,
    compare the extracted tensor against the corresponding slice of the
    source chunks and report whether they are bit-identical. Also check that
    item indices match between the extraction and the source, and that the
    revision hashes in the extraction provenance match the source chunk
    metadata. Report a per-combination summary and an overall verdict, and
    exit non-zero if anything mismatches.

Usage:
    /root/venv/bin/python scripts/check_extraction.py \
        /workspace/activations /workspace/cand_end_layers [--sample 8]
First argument: the activations ROOT holding MODEL/SET subdirectories (as
for extract_cand_end.py, NOT a leaf). Second: the directory of
layer_{L:02d}.pt files that extract_cand_end.py wrote. --sample N checks N
evenly spaced layers instead of all of them.

I/O structure: each source MODEL/SET leaf is read once via
extract_cand_end.read_model_set (the same pass the extraction itself used,
keeping only the cand_end slice, ~6.3 GiB RAM at full scale); each sampled
layer file is then loaded once and all six combinations are compared against
it. Bit-identity is torch.equal on the bf16 tensors — the extraction is a
pure slice-and-save, so any difference at all is a failure.
"""

import argparse
import sys
from pathlib import Path

import torch

from cache_activations import MODELS, SETS
from extract_cand_end import CAND_END_IDX, read_model_set


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify extract_cand_end.py output against source chunks"
    )
    parser.add_argument(
        "activations_root", type=Path,
        help="activations ROOT holding MODEL/SET subdirectories (as for "
             "extract_cand_end.py, not a MODEL/SET leaf)")
    parser.add_argument(
        "extraction_dir", type=Path,
        help="directory of layer_{L:02d}.pt files written by extract_cand_end.py")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="check only this many evenly spaced layers (default: all)")
    args = parser.parse_args()

    failures = []

    # ---- read sources (one pass per leaf, cand_end slice only) -------------
    sources, source_idx, source_rev = {}, {}, {}
    for set_key in SETS:
        for model_key in MODELS:
            leaf = args.activations_root / model_key / set_key
            if not leaf.is_dir():
                sys.exit(f"missing source directory {leaf}; the first "
                         f"argument must be the activations ROOT")
            acts, idx, meta, _ = read_model_set(leaf)
            sources[(set_key, model_key)] = acts
            source_idx[(set_key, model_key)] = idx
            source_rev[model_key] = meta["revision"]
            print(f"read source {leaf}: {acts.shape[0]} items", flush=True)

    n_layers = next(iter(sources.values())).shape[2]
    layer_files = sorted(args.extraction_dir.glob("layer_*.pt"))
    if not layer_files:
        sys.exit(f"no layer_*.pt files in {args.extraction_dir}")
    if len(layer_files) != n_layers:
        failures.append(f"extraction has {len(layer_files)} layer files, "
                        f"sources have {n_layers} layers")

    if args.sample is not None and args.sample < n_layers:
        if args.sample <= 1:
            picked = [n_layers // 2]
        else:
            picked = sorted({round(i * (n_layers - 1) / (args.sample - 1))
                             for i in range(args.sample)})
    else:
        picked = list(range(n_layers))
    print(f"checking {len(picked)} layer(s): {picked}", flush=True)

    # ---- compare ------------------------------------------------------------
    mismatched = {key: [] for key in sources}   # (set, model) -> bad layers
    idx_ok = {key: True for key in sources}
    rev_ok = {model_key: True for model_key in MODELS}

    for layer in picked:
        path = args.extraction_dir / f"layer_{layer:02d}.pt"
        if not path.exists():
            failures.append(f"missing {path.name}")
            continue
        d = torch.load(path)
        meta = d["meta"]
        if meta.get("layer") != layer:
            failures.append(f"{path.name}: meta['layer'] is "
                            f"{meta.get('layer')}, expected {layer}")
        if (meta.get("position_index") != CAND_END_IDX
                or meta.get("position_name") != "cand_end"):
            failures.append(f"{path.name}: position provenance is not "
                            f"cand_end / index {CAND_END_IDX}")
        for model_key in MODELS:
            if meta["revisions"].get(model_key) != source_rev[model_key]:
                rev_ok[model_key] = False
        for set_key in SETS:
            for model_key in MODELS:
                key = (set_key, model_key)
                extracted = d["acts"][set_key][model_key]
                src = sources[key][:, :, layer, :]
                if not torch.equal(extracted, src):
                    mismatched[key].append(layer)
                if d["item_indices"][set_key] != source_idx[key]:
                    idx_ok[key] = False

    # ---- report -------------------------------------------------------------
    print(f"\n=== Per-combination summary "
          f"({len(picked)} layer(s) checked) ===")
    print(f"{'set':<9} {'model':<19} {'bit-identical':<28} {'item indices'}")
    for set_key in SETS:
        for model_key in MODELS:
            key = (set_key, model_key)
            bad = mismatched[key]
            shown = bad if len(bad) <= 10 else bad[:10] + ["..."]
            ident = "yes" if not bad else f"NO, layers {shown}"
            idx_str = "match" if idx_ok[key] else "MISMATCH"
            print(f"{set_key:<9} {model_key:<19} {ident:<28} {idx_str}")
            if bad:
                failures.append(f"{set_key}/{model_key}: tensors differ at "
                                f"layers {bad}")
            if not idx_ok[key]:
                failures.append(f"{set_key}/{model_key}: item indices differ")
    print("revisions:", "  ".join(
        f"{model_key}={'match' if ok else 'MISMATCH'}"
        for model_key, ok in rev_ok.items()))
    for model_key, ok in rev_ok.items():
        if not ok:
            failures.append(f"{model_key}: revision hash in extraction "
                            f"provenance differs from source chunks")

    print("\n=== Verdict ===")
    if failures:
        print(f"FAIL: {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(f"PASS: extraction is bit-identical to the cand_end slice of the "
          f"source chunks at all {len(picked)} checked layer(s), item "
          f"indices and revision hashes match, for all "
          f"{len(SETS)}x{len(MODELS)} combinations.")


if __name__ == "__main__":
    main()
