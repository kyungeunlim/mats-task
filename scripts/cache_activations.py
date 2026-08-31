"""Activation caching for the probe study (plan.md T4).

Prompt (2026-08-31, implements the caching step in docs/plan.md T4):
    Write scripts/cache_activations.py implementing the activation caching in
    plan.md T4. Read plan.md T2 and T4 and results.md T4/T4b first, and reuse
    the loading pattern from scripts/check_tl_vs_hf.py, which is verified
    against HF forward passes in both dtypes.

    For each of the three checkpoints (EleutherAI/deep-ignorance-unfiltered,
    EleutherAI/deep-ignorance-e2e-strong-filter,
    kelim/deep-ignorance-unlearned-cb) and each of the two item sets
    (data/main_set.json and the control set), run four forward passes per
    item, one per candidate answer, building the prompt exactly as
    check_tl_vs_hf.py builds it. Cache the residual stream after every one of
    the 32 blocks at three token positions: end of the question text, the
    "Answer:" marker, and end of the candidate answer. Locate those positions
    from the tokenized prompt rather than assuming offsets. bf16 throughout.
    One model at a time, freeing it before loading the next. Write to the
    volume under /workspace, not container disk.

    Loading requirements carried from the check script and results.md: guard
    torch.backends.mps before importing transformer_lens; load via local
    snapshot paths, or set HF_HUB_OFFLINE only after models are loaded; alias
    hf_model.embed_out to hf_model.lm_head for transformers 5.x; load TL with
    from_pretrained_no_processing("EleutherAI/pythia-6.9b", hf_model=...,
    tokenizer=..., dtype=torch.bfloat16) and assert the resulting TL config
    matches the HF config on n_layers, d_model, and d_vocab before caching.

    Add a --limit N flag so a small subset can be run first, and a --model
    flag to run one checkpoint. Batch the four candidates of an item together
    if that's straightforward; make batch size a flag either way. Print items
    per second and estimated total time. Write progress to stdout so the run
    can be followed from a log file.

    The run must be restartable: write per-model, per-item-set output files
    and skip work already on disk, so an interrupted run doesn't start over.

    Print a per-position norm check per model with position 0 shown
    separately, since the smoke test found position 0 carries much larger
    norms. For the first few items, print the tokens at the three chosen
    positions so they can be checked by eye against the prompt.

    Record in the output file, alongside the tensors: the checkpoint revision
    hash, dtype, TL and torch versions, the item-set file and its seed, and
    the position indices used. Write in chunks rather than one file per item,
    since the volume has high small-file latency. One file per model per item
    set, or per chunk of items if that makes restarting simpler. Print
    cumulative bytes written as the run progresses.

2026-08-31, appended prompt (weights_only-safe metadata):
    In scripts/cache_activations.py, the chunk .pt files can't be loaded
    with torch.load's default weights_only=True: it fails on a
    torch.torch_version.TorchVersion object in the meta dict. torch.version
    is a TorchVersion, which subclasses str and prints like one, so wrapping
    it in str() isn't automatic. Coerce every metadata value to a plain str,
    int, float, list, or dict before saving, so the chunks load under the
    safe default. T5 will be reading these files. Check the transformers and
    transformer_lens version values too, and confirm the fix by writing a
    chunk and loading it back without weights_only=False.
    [Checked: only torch.__version__ is a str subclass; the
    importlib.metadata version strings are plain str and the config values
    plain int. provenance() is coerced recursively anyway.]

Implementation notes:
- Loading: HF checkpoints are loaded from their local snapshot paths under
  $HF_HOME/hub (revision hash read from refs/main and recorded), so
  HF_HUB_OFFLINE is never needed for them. TL still resolves the pythia-6.9b
  name against the cached config, as in check_tl_vs_hf.py.
- The mps guard: results.md T4 note 1 says check_tl_vs_hf.py carries a
  runtime guard for TL 3.7.3 touching torch.backends.mps at import, but no
  such guard was ever committed (it was a pod-local fix). The guard here is
  implemented from the results.md description: stub torch.backends.mps with
  is_available/is_built returning False if the torch build lacks it. On the
  current pod build (torch 2.13.0+cu130) the attribute exists and the guard
  is a no-op; it has not been exercised on a build where it fires.
- Positions are located with the fast tokenizer's offset_mapping: each cached
  position is the index of the token containing the LAST character of its
  region (question text / "Answer:" / candidate). Verified on a sample prompt
  to land on '?', ':', and the final token. The candidate-end position is
  asserted to be the last token of the sequence.
- Batching: the four candidates of --batch-items items run as one batch,
  right-padded with the eos id (NeoX has no pad token). No attention mask is
  needed: attention is causal and positions are rotary, so a real token at
  index i attends only to indices <= i, which are all real tokens; the
  activations at the cached positions are therefore identical to an unpadded
  run, with pad tokens never attended to.
- Forward pass runs with stop_at_layer=n_layers (all blocks run, their
  resid_post hooks fire; ln_final and the unembed are skipped as unused).
- Output: {out_dir}/{model}/{set}/chunk_{start:05d}-{end:05d}.pt, ~400 MB at
  the default 128 items per chunk. Written atomically (.tmp then rename), so
  an existing chunk file is complete and is skipped on restart. Chunk
  boundaries depend on --limit and --chunk-items; resume with the same flags,
  otherwise leftover non-matching chunk files are reported and ignored.
  Expected volume: 1076 items x 4 candidates x 32 layers x 3 positions x
  4096 x 2 bytes ~ 3.2 GiB per model per set, ~19 GiB for all six.
- Each chunk stores, alongside the bf16 activations
  (items, 4 candidates, n_layers, 3 positions, d_model): the position-index
  tensor (items, 4, 3), sequence lengths, global item indices, and a
  provenance dict (checkpoint revision hash, dtype, torch / transformers /
  transformer_lens versions, item-set file and seeds, position names, prompt
  template). A meta.json per model/set repeats the provenance and lists the
  chunks once the set completes. A NaN check runs on every chunk before save.
- bf16 caveat carried from results.md T4b: block 31 carries roughly 1%
  relative rounding noise at typical positions in bf16.

Run on the pod (background, with a log):
    nohup /root/venv/bin/python scripts/cache_activations.py \
        > results/cache_activations.log 2>&1 &
    Smoke test first: scripts/cache_activations.py --limit 4 --model unfiltered
"""

import argparse
import gc
import json
import os
import time
import types
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

import torch

MODELS = {
    "unfiltered": "EleutherAI/deep-ignorance-unfiltered",
    "e2e-strong-filter": "EleutherAI/deep-ignorance-e2e-strong-filter",
    "unlearned-cb": "kelim/deep-ignorance-unlearned-cb",
}
SETS = {"main": "data/main_set.json", "control": "data/control_set.json"}
# TL official name whose architecture deep-ignorance shares; see
# check_tl_vs_hf.py's docstring for the hf_model= loading pattern.
TL_OFFICIAL_NAME = "EleutherAI/pythia-6.9b"
DTYPE = torch.bfloat16
DTYPE_NAME = "bfloat16"

REPO_ROOT = Path(__file__).resolve().parent.parent

# Prompt pieces, verbatim from lm_eval_tasks/wmdp_bio_cloze_verified/ (same
# constants as check_tl_vs_hf.py, which verified the resulting forward pass).
DESCRIPTION = "Complete the following biology questions with the correct answer.\n\n"
DOC_TO_TEXT = "Question: {question}\nAnswer:"
TARGET_DELIMITER = " "  # lm-eval default for multiple_choice continuations

POSITION_NAMES = ("q_end", "answer_marker", "cand_end")
N_CANDIDATES = 4


def guard_mps() -> None:
    # TL 3.7.3 calls torch.backends.mps.is_available() at import time
    # (results.md T4 note 1); stub it out if this torch build lacks it.
    if not hasattr(torch.backends, "mps"):
        torch.backends.mps = types.SimpleNamespace(
            is_available=lambda: False, is_built=lambda: False
        )


def resolve_snapshot(repo_id: str) -> tuple[Path, str]:
    hf_home = Path(os.environ.get("HF_HOME", "/workspace/hf"))
    model_dir = hf_home / "hub" / ("models--" + repo_id.replace("/", "--"))
    rev = (model_dir / "refs" / "main").read_text().strip()
    snap = model_dir / "snapshots" / rev
    assert snap.is_dir(), f"no local snapshot for {repo_id} at {snap}"
    return snap, rev


def token_containing(offsets, char_end: int) -> int:
    # Index of the token containing the character at char_end - 1, i.e. the
    # last character of the region ending at char_end.
    for i, (start, end) in enumerate(offsets):
        if start < char_end <= end:
            return i
    raise ValueError(f"no token contains char {char_end - 1}")


def encode_item(tokenizer, item) -> tuple[list[list[int]], list[list[int]]]:
    """Tokenize the four candidate prompts of one item.

    Returns (ids_per_candidate, positions_per_candidate) where positions are
    [q_end, answer_marker, cand_end] token indices, located from the
    offset mapping of each full prompt rather than assumed.
    """
    question = item["question"].strip()
    stem = DESCRIPTION + DOC_TO_TEXT.format(question=question)
    q_end_char = len(DESCRIPTION) + len("Question: ") + len(question)
    marker_end_char = len(stem)  # end of "Answer:"
    ids_list, pos_list = [], []
    for choice in item["choices"]:
        prompt = stem + TARGET_DELIMITER + choice
        enc = tokenizer(prompt, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        pos = [
            token_containing(offsets, q_end_char),
            token_containing(offsets, marker_end_char),
            token_containing(offsets, len(prompt)),
        ]
        assert pos[2] == len(enc["input_ids"]) - 1, (
            f"cand_end is not the last token: {pos[2]} vs {len(enc['input_ids']) - 1}"
        )
        ids_list.append(enc["input_ids"])
        pos_list.append(pos)
    return ids_list, pos_list


def print_token_check(tokenizer, items, set_key: str, n_show: int = 3) -> None:
    print(f"\n=== Token position check, {set_key} set, "
          f"first {min(n_show, len(items))} items ===", flush=True)
    for idx in range(min(n_show, len(items))):
        ids_list, pos_list = encode_item(tokenizer, items[idx])
        for c, (ids, pos) in enumerate(zip(ids_list, pos_list)):
            toks = [tokenizer.decode([ids[p]]) for p in pos]
            named = "  ".join(
                f"{name}@{p}={tok!r}"
                for name, p, tok in zip(POSITION_NAMES, pos, toks)
            )
            print(f"  item {idx} cand {c} ({len(ids)} tokens): {named}")


def load_tl_model(model_key: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    guard_mps()  # must precede the transformer_lens import
    from transformer_lens import HookedTransformer

    repo_id = MODELS[model_key]
    snap, rev = resolve_snapshot(repo_id)
    print(f"\n=== Loading {repo_id} ===\n"
          f"snapshot: {snap}\nrevision: {rev}\ndtype: {DTYPE_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(snap))
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(snap), torch_dtype=DTYPE
    ).to(device)
    hf_model.eval()
    hf_cfg = hf_model.config
    print(f"HF  num_hidden_layers={hf_cfg.num_hidden_layers}  "
          f"hidden_size={hf_cfg.hidden_size}  vocab_size={hf_cfg.vocab_size}",
          flush=True)

    # transformers 5.x renamed the NeoX LM head embed_out -> lm_head; TL
    # 3.7.3's convert_neox_weights still reads .embed_out (shared weights).
    if not hasattr(hf_model, "embed_out"):
        hf_model.embed_out = hf_model.lm_head

    tl_model = HookedTransformer.from_pretrained_no_processing(
        TL_OFFICIAL_NAME,
        hf_model=hf_model,
        tokenizer=tokenizer,
        dtype=DTYPE,
        device=device,
    )
    tl_model.eval()
    cfg = tl_model.cfg
    print(f"TL  n_layers={cfg.n_layers}  d_model={cfg.d_model}  "
          f"d_vocab={cfg.d_vocab}  (from_pretrained_no_processing: no folding "
          f"or centering)", flush=True)
    assert cfg.n_layers == hf_cfg.num_hidden_layers, "n_layers mismatch"
    assert cfg.d_model == hf_cfg.hidden_size, "d_model mismatch"
    assert cfg.d_vocab == hf_cfg.vocab_size, "d_vocab mismatch"

    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    return tl_model, tokenizer, rev


def run_batch(tl_model, ids_list, pos_list, pad_id, device, first_batch):
    """One forward pass over len(ids_list) right-padded sequences.

    Returns (n_seqs, n_layers, 3, d_model) bf16 on CPU. Right padding with no
    attention mask is exact here: causal attention plus rotary positions mean
    a real token attends only to earlier real tokens (see module docstring).
    """
    cfg = tl_model.cfg
    n_seqs = len(ids_list)
    max_len = max(len(x) for x in ids_list)
    batch = torch.full((n_seqs, max_len), pad_id, dtype=torch.long)
    for j, ids in enumerate(ids_list):
        batch[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    pos = torch.tensor(pos_list, dtype=torch.long, device=device)  # (n_seqs, 3)
    rows = torch.arange(n_seqs, device=device)[:, None]

    hooks = {f"blocks.{i}.hook_resid_post" for i in range(cfg.n_layers)}
    with torch.no_grad():
        _, cache = tl_model.run_with_cache(
            batch.to(device),
            names_filter=lambda name: name in hooks,
            stop_at_layer=cfg.n_layers,  # all blocks run; skips ln_final/unembed
        )
    gathered, norm_rows = [], []
    for layer in range(cfg.n_layers):
        resid = cache[f"blocks.{layer}.hook_resid_post"]  # (n_seqs, max_len, d)
        picked = resid[rows, pos]  # (n_seqs, 3, d_model)
        gathered.append(picked.cpu())
        if first_batch:
            norm_rows.append(
                [resid[:, 0, :].float().norm(dim=-1).mean().item()]
                + picked.float().norm(dim=-1).mean(dim=0).tolist()
            )
    del cache
    return torch.stack(gathered, dim=1), norm_rows  # (n_seqs, n_layers, 3, d)


def print_norm_check(norm_rows, model_key: str, set_key: str) -> None:
    print(f"\n=== Per-position norm check, {model_key} / {set_key} "
          f"(mean L2 over first batch; token position 0 shown separately, "
          f"it carries much larger norms and is not among the cached "
          f"positions) ===")
    header = f"{'layer':>5} {'pos0':>10} " + " ".join(
        f"{n:>13}" for n in POSITION_NAMES
    )
    print(header)
    for layer, row in enumerate(norm_rows):
        print(f"{layer:>5} {row[0]:>10.1f} "
              + " ".join(f"{v:>13.1f}" for v in row[1:]))


def fmt_secs(s: float) -> str:
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def fmt_bytes(b: float) -> str:
    return f"{b / 2**20:.1f} MiB" if b < 2**30 else f"{b / 2**30:.2f} GiB"


def plain(value):
    """Coerce metadata to plain str/int/float/bool/None/list/dict so chunk
    files load under torch.load's default weights_only=True. Needed because
    torch.__version__ is a TorchVersion, a str subclass the safe unpickler
    rejects; applied recursively so no future metadata value can regress it.
    """
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value)


def provenance(model_key, rev, set_key, set_meta, cfg) -> dict:
    return plain({
        "model": MODELS[model_key],
        "revision": rev,
        "tl_official_name": TL_OFFICIAL_NAME,
        "dtype": DTYPE_NAME,
        "torch_version": torch.__version__,
        "transformers_version": pkg_version("transformers"),
        "transformer_lens_version": pkg_version("transformer_lens"),
        "item_set_file": SETS[set_key],
        "item_set_seeds": {
            k: set_meta[k]
            for k in ("seed", "sampling_seed", "split_seed") if k in set_meta
        },
        "n_layers": cfg.n_layers,
        "d_model": cfg.d_model,
        "position_names": list(POSITION_NAMES),
        "prompt_template": {
            "description": DESCRIPTION,
            "doc_to_text": DOC_TO_TEXT,
            "target_delimiter": TARGET_DELIMITER,
        },
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })


def cache_set(tl_model, tokenizer, model_key, set_key, rev, args, stats) -> None:
    set_json = json.loads((REPO_ROOT / SETS[set_key]).read_text())
    items = set_json["items"]
    n_items = min(len(items), args.limit) if args.limit else len(items)
    out_dir = Path(args.out_dir) / model_key / set_key
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = tl_model.cfg
    meta = provenance(model_key, rev, set_key, set_json, cfg)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id  # NeoX has no pad token; see docstring

    print_token_check(tokenizer, items, set_key)

    expected = {
        (s, min(s + args.chunk_items, n_items)):
            out_dir / f"chunk_{s:05d}-{min(s + args.chunk_items, n_items):05d}.pt"
        for s in range(0, n_items, args.chunk_items)
    }
    stray = sorted(
        p.name for p in out_dir.glob("chunk_*.pt")
        if p not in expected.values()
    )
    if stray:
        print(f"WARNING: {out_dir} holds chunk files not matching the current "
              f"--limit/--chunk-items boundaries, ignored: {stray}", flush=True)

    first_batch = True
    for (start, end), path in expected.items():
        if path.exists():
            print(f"[{model_key}/{set_key}] chunk {start}-{end} exists, "
                  f"skipping ({fmt_bytes(path.stat().st_size)})", flush=True)
            stats["skipped_items"] += end - start
            continue

        acts = torch.empty(
            (end - start, N_CANDIDATES, cfg.n_layers, 3, cfg.d_model),
            dtype=DTYPE,
        )
        positions = torch.empty((end - start, N_CANDIDATES, 3), dtype=torch.long)
        seq_lens = torch.empty((end - start, N_CANDIDATES), dtype=torch.long)

        for b_start in range(start, end, args.batch_items):
            b_end = min(b_start + args.batch_items, end)
            ids_list, pos_list = [], []
            for idx in range(b_start, b_end):
                ids, pos = encode_item(tokenizer, items[idx])
                ids_list.extend(ids)
                pos_list.extend(pos)
            t0 = time.monotonic()
            gathered, norm_rows = run_batch(
                tl_model, ids_list, pos_list, pad_id, args.device, first_batch
            )
            stats["forward_secs"] += time.monotonic() - t0
            if first_batch:
                print_norm_check(norm_rows, model_key, set_key)
                first_batch = False

            k = b_end - b_start
            sl = slice(b_start - start, b_end - start)
            acts[sl] = gathered.view(k, N_CANDIDATES, cfg.n_layers, 3, cfg.d_model)
            positions[sl] = torch.tensor(pos_list).view(k, N_CANDIDATES, 3)
            seq_lens[sl] = torch.tensor(
                [len(x) for x in ids_list]).view(k, N_CANDIDATES)

            stats["done_items"] += k
            rate = stats["done_items"] / max(stats["forward_secs"], 1e-9)
            remaining = stats["total_items"] - stats["done_items"] - stats["skipped_items"]
            print(f"[{model_key}/{set_key}] items {b_end}/{n_items}  "
                  f"{rate:.2f} items/s  est. remaining (all selected runs, "
                  f"excl. model loads) {fmt_secs(remaining / rate)}  "
                  f"cumulative written {fmt_bytes(stats['bytes'])}", flush=True)

        assert not torch.isnan(acts.float()).any(), (
            f"NaNs in chunk {start}-{end} of {model_key}/{set_key}"
        )
        payload = {
            "acts": acts,  # (items, candidate, layer, position, d_model) bf16
            "positions": positions,  # token indices, (items, candidate, 3)
            "seq_lens": seq_lens,
            "item_indices": list(range(start, end)),
            "meta": meta,
        }
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        size = path.stat().st_size
        stats["bytes"] += size
        print(f"[{model_key}/{set_key}] wrote {path.name} "
              f"({fmt_bytes(size)}, cumulative {fmt_bytes(stats['bytes'])})",
              flush=True)

    (out_dir / "meta.json").write_text(json.dumps(
        meta | {"n_items": n_items, "chunks": [p.name for p in expected.values()]},
        indent=2,
    ))
    print(f"[{model_key}/{set_key}] set complete, meta.json written", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache residual-stream activations (plan.md T4)"
    )
    parser.add_argument("--model", choices=list(MODELS), default=None,
                        help="run one checkpoint (default: all three)")
    parser.add_argument("--set", choices=list(SETS), default=None,
                        help="run one item set (default: both)")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap items per set, for a small first run")
    parser.add_argument("--batch-items", type=int, default=8,
                        help="items per forward batch (4 sequences each)")
    parser.add_argument("--chunk-items", type=int, default=128,
                        help="items per output file; resume skips whole chunks")
    parser.add_argument("--out-dir", default="/workspace/activations",
                        help="output root on the volume")
    args = parser.parse_args()
    args.device = "cuda"
    assert torch.cuda.is_available(), "this script is meant to run on the pod GPU"

    model_keys = [args.model] if args.model else list(MODELS)
    set_keys = [args.set] if args.set else list(SETS)

    per_set = {}
    for set_key in set_keys:
        n = len(json.loads((REPO_ROOT / SETS[set_key]).read_text())["items"])
        per_set[set_key] = min(n, args.limit) if args.limit else n
    stats = {
        "total_items": sum(per_set.values()) * len(model_keys),
        "done_items": 0, "skipped_items": 0,
        "forward_secs": 0.0, "bytes": 0,
    }
    print(f"Caching {stats['total_items']} items total: models {model_keys}, "
          f"sets {per_set}, {N_CANDIDATES} passes per item, dtype {DTYPE_NAME}, "
          f"out dir {args.out_dir}", flush=True)

    for model_key in model_keys:
        tl_model, tokenizer, rev = load_tl_model(model_key, args.device)
        for set_key in set_keys:
            cache_set(tl_model, tokenizer, model_key, set_key, rev, args, stats)
        del tl_model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"=== {model_key} done, model freed ===", flush=True)

    print(f"\nAll done: {stats['done_items']} items cached this run, "
          f"{stats['skipped_items']} skipped (already on disk), "
          f"{fmt_bytes(stats['bytes'])} written, "
          f"forward time {fmt_secs(stats['forward_secs'])}", flush=True)


if __name__ == "__main__":
    main()
