"""Extract cand_end activations into per-layer files for T5 (plan.md T5 prep).

Prompt (2026-08-31, prepares T5 probe inputs from the T4 cache):
    Write scripts/extract_cand_end.py, which reads the cached chunks under an
    activations root and writes a compact per-layer extraction so T5 can run
    without the caching pod. For each layer 0 to 31, produce one file holding
    the cand_end position only (index 2 of the position axis) for all three
    models and both item sets, so T5 can load one layer at a time. Keep bf16,
    since that is what was cached. Alongside the tensors, carry the item
    indices and enough provenance to trace back: the three revision hashes,
    dtype, item set files and seeds, the source chunk filenames, and the
    position index and name extracted. Assert that item indices are
    consistent across models within a set before writing, since the probe
    compares the same items across models. Take the activations root and an
    output directory as arguments, with the same leaf-path clarity as
    check_batch_padding.py. Print per-layer file sizes and a running total.
    The files must load under torch.load's default weights_only=True, as
    with cache_activations.py.

Output: OUT_DIR/layer_{L:02d}.pt for L in 0..n_layers-1, each a dict
    {"acts": {set: {model: bf16 tensor (n_items, 4 candidates, d_model)}},
     "item_indices": {set: [global item indices]},
     "meta": provenance}
loaded with plain torch.load (weights_only default). All metadata is coerced
through cache_activations.plain(), so the safe unpickler accepts it. Layer
slices are made contiguous before saving so each file holds only its own
layer, not a view of the full storage.

Usage:
    /root/venv/bin/python scripts/extract_cand_end.py \
        /workspace/activations /workspace/cand_end_layers
The first argument is the activations ROOT that cache_activations.py's
--out-dir produced (the directory containing MODEL/SET subdirectories) — not
a MODEL/SET leaf directory like check_batch_padding.py takes. All three
models and both sets must be present under it. The second argument is the
output directory for the per-layer files (created if missing).
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from cache_activations import MODELS, SETS, plain
from check_batch_padding import POSITION_NAMES

CAND_END_IDX = 2  # index of cand_end on the position axis of cached chunks


def fmt_bytes(b: float) -> str:
    return f"{b / 2**20:.1f} MiB" if b < 2**30 else f"{b / 2**30:.2f} GiB"


def read_model_set(leaf: Path):
    """Load every chunk of one MODEL/SET leaf, keeping only cand_end.

    Returns (acts (items, cand, layers, d_model) bf16, item_indices,
    first chunk meta, chunk filenames).
    """
    files = sorted(leaf.glob("chunk_*.pt"))
    if not files:
        sys.exit(f"no chunk_*.pt files in {leaf}; the first argument must be "
                 f"the activations ROOT holding MODEL/SET subdirectories "
                 f"with chunks for all of models {list(MODELS)} and sets "
                 f"{list(SETS)}")
    acts_parts, indices, names = [], [], []
    meta0 = None
    for path in files:
        chunk = torch.load(path)  # weights_only default; sources must be safe
        meta = chunk["meta"]
        if meta0 is None:
            meta0 = meta
        assert meta["revision"] == meta0["revision"], f"{path}: revision differs"
        assert meta["dtype"] == meta0["dtype"], f"{path}: dtype differs"
        acts_parts.append(chunk["acts"][:, :, :, CAND_END_IDX, :])
        indices.extend(chunk["item_indices"])
        names.append(path.name)
    assert meta0["dtype"] == "bfloat16", f"{leaf}: cached dtype is not bfloat16"
    assert meta0["position_names"][CAND_END_IDX] == POSITION_NAMES[CAND_END_IDX] == "cand_end", (
        f"{leaf}: position axis index {CAND_END_IDX} is not cand_end"
    )
    acts = torch.cat(acts_parts)
    assert acts.dtype == torch.bfloat16
    assert indices == sorted(indices), f"{leaf}: item indices not ascending"
    assert len(indices) == acts.shape[0]
    return acts, indices, meta0, names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract cand_end activations into per-layer files for T5"
    )
    parser.add_argument(
        "activations_root", type=Path,
        help="activations ROOT produced by cache_activations.py --out-dir "
             "(contains MODEL/SET subdirectories; NOT a MODEL/SET leaf like "
             "check_batch_padding.py takes)")
    parser.add_argument(
        "out_dir", type=Path,
        help="output directory for layer_{L:02d}.pt files (created if missing)")
    args = parser.parse_args()

    missing = [f"{m}/{s}" for m in MODELS for s in SETS
               if not (args.activations_root / m / s).is_dir()]
    if missing:
        sys.exit(f"missing MODEL/SET directories under {args.activations_root}: "
                 f"{missing}; the extraction needs all three models and both "
                 f"sets")

    acts = {}          # (set_key, model_key) -> (items, cand, layers, d_model)
    item_indices = {}  # set_key -> list of global item indices
    revisions, chunk_names, set_meta = {}, {}, {}
    versions = None
    n_layers = d_model = None
    for set_key in SETS:
        chunk_names[set_key] = {}
        for model_key in MODELS:
            leaf = args.activations_root / model_key / set_key
            a, idx, meta, names = read_model_set(leaf)
            print(f"read {leaf}: {a.shape[0]} items, {len(names)} chunk(s)",
                  flush=True)
            # The probe compares the same items across models, so the three
            # models of a set must cover identical item indices in order.
            if set_key not in item_indices:
                item_indices[set_key] = idx
            else:
                assert idx == item_indices[set_key], (
                    f"{leaf}: item indices differ from the other models of "
                    f"the {set_key} set; the caches do not cover the same items"
                )
            if model_key not in revisions:
                revisions[model_key] = meta["revision"]
            else:
                assert revisions[model_key] == meta["revision"], (
                    f"{leaf}: revision differs from the other set of {model_key}"
                )
            if n_layers is None:
                n_layers, d_model = a.shape[2], a.shape[3]
            assert (a.shape[2], a.shape[3]) == (n_layers, d_model), (
                f"{leaf}: n_layers/d_model differ across caches"
            )
            acts[(set_key, model_key)] = a
            chunk_names[set_key][model_key] = names
            set_meta[set_key] = {"file": meta["item_set_file"],
                                 "seeds": meta["item_set_seeds"]}
            versions = {k: meta[k] for k in
                        ("torch_version", "transformers_version",
                         "transformer_lens_version")}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_meta = plain({
        "position_index": CAND_END_IDX,
        "position_name": "cand_end",
        "dtype": "bfloat16",
        "models": dict(MODELS),
        "revisions": revisions,
        "item_sets": set_meta,
        "source_chunks": chunk_names,
        "source_versions": versions,
        "n_layers": n_layers,
        "d_model": d_model,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })

    total = 0
    for layer in range(n_layers):
        payload = {
            "acts": {
                set_key: {
                    model_key: acts[(set_key, model_key)][:, :, layer, :].contiguous()
                    for model_key in MODELS
                } for set_key in SETS
            },
            "item_indices": {k: list(v) for k, v in item_indices.items()},
            "meta": base_meta | {"layer": layer},
        }
        path = args.out_dir / f"layer_{layer:02d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        size = path.stat().st_size
        total += size
        print(f"wrote {path.name}  {fmt_bytes(size)}  (running total "
              f"{fmt_bytes(total)})", flush=True)

    print(f"\nDone: {n_layers} layer files in {args.out_dir}, "
          f"{fmt_bytes(total)} total", flush=True)


if __name__ == "__main__":
    main()
