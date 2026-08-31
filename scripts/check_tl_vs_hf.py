"""Pre-caching numerical check: TransformerLens vs HF forward pass (plan.md T4).

Prompt (2026-08-28, implements the [rev] pre-caching check in docs/plan.md T4):
    Write scripts/check_tl_vs_hf.py implementing the pre-caching check in
    plan.md T4. Load EleutherAI/deep-ignorance-unfiltered twice: once via
    HookedTransformer.from_pretrained_no_processing and once via plain HF
    AutoModelForCausalLM, both bfloat16, on the GPU. First print the config's
    layer count and hidden size. Take item 0 from data/main_set.json, build the
    prompt exactly as the eval does (the description line, then
    "Question: {question}\\nAnswer:" with the correct candidate appended),
    tokenize once, and run it through both models. Compare the final-layer
    residual stream from TL against the HF hidden state at the same depth, and
    report the max absolute difference and whether it is within a stated
    bfloat16 tolerance. Also print the TL hook name used and the tensor shapes.
    If the two disagree, say where they first diverge by layer rather than
    just failing.

2026-08-28, after first run: TL does not recognize deep-ignorance names
    (ValueError from get_official_model_name), so "load twice" is amended: the
    HF model is loaded once from disk and passed to TL via
    from_pretrained_no_processing("EleutherAI/pythia-6.9b", hf_model=...),
    the standard TL pattern for an unlisted checkpoint that shares a listed
    architecture. deep-ignorance-unfiltered's config matches pythia-6.9b on
    every architectural field except vocab_size (50304 vs 50432); TL overrides
    d_vocab from the passed hf_model's config (loading_from_pretrained.py,
    cfg_dict["d_vocab"] = hf_cfg.get("vocab_size", ...)), and the script
    asserts the resulting TL config against the HF one before comparing. TL
    still converts the weights into its own modules and runs its own forward
    pass, so the TL-vs-HF computation comparison is unaffected; only the disk
    load is shared. Requires pythia-6.9b's config.json in the HF cache (1KB,
    fetched once; weights are never downloaded).

2026-08-31, appended prompt (raw last-block residual via forward hook):
    In scripts/check_tl_vs_hf.py, add a raw-residual comparison for the last
    block. HF GPT-NeoX doesn't expose block 31's raw output in hidden_states
    (the final entry is post-final-LN), so register a forward hook on
    hf_model.gpt_neox.layers[n_layers - 1] before the HF forward pass, storing
    output[0] if the output is a tuple else output, as float32 on CPU. After
    the loop over blocks 0 to n-2, add a report_row for
    blocks.{last}.hook_resid_post against the hooked tensor, using the same
    4-ULP criterion, and set first_divergence to
    "layer {last} (via HF forward hook)" if it fails. Leave the post-final-LN
    row and its threshold unchanged, that decision waits for this result. Also
    update the memory note in the docstring: the current pod is an RTX PRO
    6000 with 96GB, so the 28GB peak has headroom.

2026-08-31, appended prompt (blocks.31 contraction diagnostics):
    The blocks.31 row fails at 6.4 ULPs, but max|HF| drops from 3856 at block
    30 to 644 at block 31, so the residual contracts about six-fold in the
    last block. Add two diagnostics for that row only, printed after it, to
    test whether that explains the failure. First, a per-position breakdown:
    for blocks.31, print the max abs diff per token position and which
    position holds the overall max, plus that position's max|HF| at block 31
    and at block 30. Second, the same block-31 diff expressed in ULPs of the
    input magnitude (block 30's output at the same position) rather than the
    output magnitude. Don't change the pass/fail criterion for any row, these
    are diagnostics only. Print position 0 separately from the rest, since
    the smoke test found position 0 carries much larger norms than other
    positions.

2026-08-31, appended prompt (--dtype flag for an fp32 control run):
    Add a way to run the whole check in float32 instead of bfloat16, so the
    blocks.31 result can be tested for whether it's purely bf16 rounding. A
    --dtype command-line flag defaulting to bfloat16 would work, applied to
    both the HF load and the TL load, with the ULP constant switched to the
    float32 value (2**-23) when fp32 is selected so the criterion scales
    correctly. Print the dtype in the header. Everything else stays the same,
    including all pass/fail criteria and the blocks.31 diagnostics.
    [fp32 memory note: two fp32 copies at the load peak are ~55GB, which
    still fits the 96GB pod.]

Notes on the comparison:
- Both models run in bfloat16 on the GPU. TL is loaded with
  from_pretrained_no_processing so no folding/centering touches the residual
  stream (the flags are logged below).
- The prompt is built exactly as lm-eval builds it for
  wmdp_bio_cloze_verified: description string from the vendored YAML, then
  doc_to_text "Question: {{question.strip()}}\\nAnswer:", then the default
  target_delimiter " " and the correct choice. Tokenized once with the HF
  tokenizer; the same input_ids tensor is fed to both models (this also
  bypasses TL's prepend_bos default, matching lm-eval, which adds no BOS for
  NeoX tokenizers).
- HF GPT-NeoX appends hidden_states BEFORE each block inside its layer loop
  and appends the final entry AFTER final_layer_norm. So hidden_states[i] for
  i in 1..n_layers-1 is the raw residual after block i-1, hidden_states[0] is
  the embedding, and hidden_states[n_layers] is the LAST block's output passed
  through the final LayerNorm. The raw final-layer residual therefore has no
  direct HF counterpart; we check hidden_states[-1] against TL's
  ln_final.hook_normalized, and all raw residuals hidden_states[1..n-1]
  against blocks.{i-1}.hook_resid_post.
- Tolerance (amended 2026-08-28 after the first complete run): bfloat16 has
  an 8-bit effective mantissa, so one rounding step is 2^-8 ~ 0.39% of
  magnitude (one ULP); TL and HF sum the same tensors in different orders and
  TL upcasts LayerNorm internally. A layer is called matched if
  max|TL-HF| <= 4 ULPs at that layer's max |HF| magnitude (~1.56% relative),
  diffs computed in float32. The first run used a flat 1% and flagged layer 1
  at rel 1.09e-2, which is exactly 2 ULPs at magnitude 92 (bf16 spacing 0.5
  there) with all later layers back at 1-2e-3 relative -- rounding noise, not
  a computational divergence, hence the ULP-based restatement.
- Also amended after the first complete run: TL's ln_final.hook_normalized is
  x-hat BEFORE the LayerNorm's scale/shift, while HF's hidden_states[-1]
  includes w,b. The script applies ln_final.w/b to TL's normalized value so
  the final-depth comparison is like with like.
- The post-final-LN row gets its own propagation-aware criterion instead of
  the flat ULP-of-tensor-max one: LN divides each position by its own std, so
  a k-ULP rounding difference in a residual element of magnitude up to
  max|x_p| becomes k*ULP*max|x_p|/sigma_p in the output -- large where the
  residual is spiky relative to its std (NeoX residuals are). Per position,
  the allowed diff is N_ULPS*ULP*(max|x_p|/sigma_p) for the propagated input
  rounding plus N_ULPS*ULP*max|out_p| for the LN's own arithmetic, with x
  taken from TL's final raw residual (which the earlier rows show matches HF
  to ~1 ULP). The row passes if every position's actual max diff is within
  its bound.
- Memory: the HF model stays on the GPU while TL converts its weights, so the
  peak is roughly two bf16 copies of a 6.9B model (~28GB) plus conversion
  transients. The current pod is an RTX PRO 6000 with 96GB, so this peak has
  plenty of headroom.

Run on the pod:
    /root/venv/bin/python scripts/check_tl_vs_hf.py [--dtype float32]
"""

import argparse
import gc
import json
from pathlib import Path

import torch

MODEL_NAME = "EleutherAI/deep-ignorance-unfiltered"
# TL official name whose architecture deep-ignorance shares; see docstring.
TL_OFFICIAL_NAME = "EleutherAI/pythia-6.9b"
BF16_ULP = 2.0 ** -8   # relative spacing of bfloat16 (8-bit effective mantissa)
FP32_ULP = 2.0 ** -23  # relative spacing of float32 (23-bit mantissa)
N_ULPS = 4  # matched if max|TL-HF| <= 4 ULPs at the layer's max |HF| magnitude

REPO_ROOT = Path(__file__).resolve().parent.parent

# Prompt pieces, verbatim from lm_eval_tasks/wmdp_bio_cloze_verified/
DESCRIPTION = "Complete the following biology questions with the correct answer.\n\n"
DOC_TO_TEXT = "Question: {question}\nAnswer:"
TARGET_DELIMITER = " "  # lm-eval default for multiple_choice continuations


def build_prompt() -> str:
    data = json.loads((REPO_ROOT / "data" / "main_set.json").read_text())
    item = data["items"][0]
    correct = item["choices"][item["answer"]]
    prompt = (
        DESCRIPTION
        + DOC_TO_TEXT.format(question=item["question"].strip())
        + TARGET_DELIMITER
        + correct
    )
    print("=== Prompt (item 0 of data/main_set.json, correct candidate) ===")
    print(repr(prompt))
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TL vs HF forward-pass check (see module docstring)"
    )
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float32"], default="bfloat16",
        help="dtype for both the HF and TL loads; the ULP constant switches "
             "with it so the criterion scales correctly",
    )
    args = parser.parse_args()
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    ulp = {"bfloat16": BF16_ULP, "float32": FP32_ULP}[args.dtype]

    assert torch.cuda.is_available(), "this check is meant to run on the pod GPU"
    device = "cuda"

    prompt = build_prompt()

    # ---- HF -----------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading {MODEL_NAME} via AutoModelForCausalLM (dtype={args.dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype
    ).to(device)
    hf_model.eval()
    hf_cfg = hf_model.config
    print("=== Config ===")
    print(f"dtype: {args.dtype}")
    print(f"HF  num_hidden_layers={hf_cfg.num_hidden_layers}  "
          f"hidden_size={hf_cfg.hidden_size}  vocab_size={hf_cfg.vocab_size}")
    n_layers = hf_cfg.num_hidden_layers

    # Tokenize ONCE with the HF tokenizer; feed the same ids to both models.
    # Passing a tensor to TL bypasses prepend_bos, matching lm-eval (no BOS).
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    print(f"\ninput_ids shape: {tuple(input_ids.shape)} "
          f"({input_ids.shape[1]} tokens, no BOS added)")

    # The last block's raw output has no hidden_states entry (the final entry
    # is post-final-LN), so capture it with a forward hook on the last layer.
    hf_last_resid = None

    def capture_last_block(module, args, output):
        nonlocal hf_last_resid
        raw = output[0] if isinstance(output, tuple) else output
        hf_last_resid = raw.float().cpu()

    hook_handle = hf_model.gpt_neox.layers[n_layers - 1].register_forward_hook(
        capture_last_block
    )

    with torch.no_grad():
        out = hf_model(input_ids, output_hidden_states=True)
    hook_handle.remove()
    assert hf_last_resid is not None, "forward hook on the last block did not fire"
    hf_hidden = [h.float().cpu() for h in out.hidden_states]
    print(f"HF hidden_states: {len(hf_hidden)} tensors "
          f"(embeddings + {n_layers - 1} raw residuals + post-final-LN), "
          f"each {tuple(hf_hidden[-1].shape)}")
    assert len(hf_hidden) == n_layers + 1
    del out

    # ---- TransformerLens ----------------------------------------------------
    from transformer_lens import HookedTransformer

    # transformers 5.x renamed the NeoX LM head embed_out -> lm_head;
    # TL 3.7.3's convert_neox_weights still reads .embed_out. Alias it
    # (shared weights, no copy) so the converter finds it.
    if not hasattr(hf_model, "embed_out"):
        hf_model.embed_out = hf_model.lm_head

    print(f"\nLoading TL via HookedTransformer.from_pretrained_no_processing"
          f"('{TL_OFFICIAL_NAME}', hf_model=<{MODEL_NAME}>, dtype={args.dtype}) ...")
    tl_model = HookedTransformer.from_pretrained_no_processing(
        TL_OFFICIAL_NAME,
        hf_model=hf_model,
        tokenizer=tokenizer,
        dtype=dtype,
        device=device,
    )
    tl_model.eval()
    cfg = tl_model.cfg
    print(f"TL  n_layers={cfg.n_layers}  d_model={cfg.d_model}  d_vocab={cfg.d_vocab}")
    print("TL load flags: from_pretrained_no_processing => "
          "fold_ln=False, center_writing_weights=False, center_unembed=False, "
          f"fold_value_biases=False, dtype={args.dtype}")
    assert cfg.n_layers == n_layers and cfg.d_model == hf_cfg.hidden_size, (
        "TL and HF configs disagree on depth/width"
    )
    assert cfg.d_vocab == hf_cfg.vocab_size, (
        f"TL d_vocab {cfg.d_vocab} != HF vocab_size {hf_cfg.vocab_size}: "
        "the hf_model config override did not take"
    )

    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    last = cfg.n_layers - 1
    final_hook = f"blocks.{last}.hook_resid_post"
    ln_final_hook = "ln_final.hook_normalized"
    wanted = (
        {"hook_embed", ln_final_hook}
        | {f"blocks.{i}.hook_resid_post" for i in range(cfg.n_layers)}
    )
    with torch.no_grad():
        _, cache = tl_model.run_with_cache(
            input_ids, names_filter=lambda name: name in wanted
        )
    tl_resids = [cache[f"blocks.{i}.hook_resid_post"].float().cpu()
                 for i in range(cfg.n_layers)]
    tl_embed = cache["hook_embed"].float().cpu()
    # hook_normalized is x-hat before the LN's scale/shift; HF's final hidden
    # state includes w,b, so apply them here to compare like with like.
    ln = tl_model.ln_final
    tl_ln_final = (cache[ln_final_hook] * ln.w + ln.b).float().cpu()
    del cache
    print(f"TL final-layer hook: {final_hook}, shape {tuple(tl_resids[-1].shape)}")

    del tl_model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Comparison ---------------------------------------------------------
    def diff_stats(a: torch.Tensor, b: torch.Tensor):
        max_abs = (a - b).abs().max().item()
        scale = b.abs().max().item()
        ulps = max_abs / (scale * ulp) if scale > 0 else 0.0
        return max_abs, scale, ulps, ulps <= N_ULPS

    def report_row(label: str, a: torch.Tensor, b: torch.Tensor) -> bool:
        max_abs, scale, ulps, ok = diff_stats(a, b)
        marker = "" if ok else "  <-- diverges"
        print(f"{label:>28} {max_abs:12.4e} {scale:10.3f} "
              f"{max_abs / scale:9.2e} {ulps:6.1f}  {ok}{marker}")
        return ok

    print(f"\n=== Layer-by-layer comparison, dtype={args.dtype} "
          f"(matched if <= {N_ULPS} {args.dtype} ULPs of max |HF|, "
          f"i.e. rel <= {N_ULPS * ulp:.2e}) ===")
    print(f"{'depth':>28} {'max|TL-HF|':>12} {'max|HF|':>10} {'rel':>9} {'ULPs':>6}  ok")
    first_divergence = None

    if not report_row("embeddings (hook_embed)", tl_embed, hf_hidden[0]):
        first_divergence = "embeddings"

    # hf_hidden[i] (1 <= i <= n_layers-1) is the raw residual after block i-1.
    for i in range(1, cfg.n_layers):
        ok = report_row(f"blocks.{i-1}.hook_resid_post", tl_resids[i - 1], hf_hidden[i])
        if not ok and first_divergence is None:
            first_divergence = f"layer {i - 1}"

    # Raw output of the last block, captured by the HF forward hook above;
    # same flat ULP criterion as the other raw-residual rows.
    ok = report_row(f"blocks.{last}.hook_resid_post", tl_resids[last], hf_last_resid)
    if not ok and first_divergence is None:
        first_divergence = f"layer {last} (via HF forward hook)"

    # Diagnostics for the blocks.{last} row only (pass/fail above unchanged):
    # the residual contracts sharply in the last block, so a diff of several
    # ULPs of the OUTPUT magnitude may still be ~1 ULP of the INPUT magnitude
    # (block {last-1}'s output). Position 0 is shown separately because it
    # carries much larger norms than the other positions.
    diff31 = (tl_resids[last][0] - hf_last_resid[0]).abs().max(dim=-1).values
    out_mag = hf_last_resid[0].abs().max(dim=-1).values
    in_mag = hf_hidden[n_layers - 1][0].abs().max(dim=-1).values  # block {last-1} out
    ulps_out31 = diff31 / (out_mag * ulp)
    ulps_in31 = diff31 / (in_mag * ulp)
    seq_len = diff31.shape[0]
    print(f"{'':>28} -- blocks.{last} diagnostics (criterion unchanged) --")
    print(f"{'':>28} position 0: diff {diff31[0].item():.4e}  "
          f"max|HF| out {out_mag[0].item():.1f} / in {in_mag[0].item():.1f}  "
          f"ULPs out {ulps_out31[0].item():.1f} / in {ulps_in31[0].item():.1f}")
    print(f"{'':>28} per-position max|TL-HF|, positions 1..{seq_len - 1}:")
    for start in range(1, seq_len, 8):
        chunk = diff31[start:start + 8].tolist()
        print(f"{'':>28}   pos {start:3d}..{min(start + 7, seq_len - 1):3d}: "
              + "  ".join(f"{v:.2e}" for v in chunk))
    p = int(diff31.argmax().item())
    print(f"{'':>28} overall max at position {p}: diff {diff31[p].item():.4e}  "
          f"max|HF| out {out_mag[p].item():.1f} (block {last}) / "
          f"in {in_mag[p].item():.1f} (block {last - 1})  "
          f"ULPs out {ulps_out31[p].item():.1f} / in {ulps_in31[p].item():.1f}")
    p1 = int(diff31[1:].argmax().item()) + 1
    print(f"{'':>28} worst excluding position 0: position {p1}: "
          f"diff {diff31[p1].item():.4e}  "
          f"ULPs out {ulps_out31[p1].item():.1f} / in {ulps_in31[p1].item():.1f}")
    print(f"{'':>28} worst ULPs-of-input over all positions: "
          f"{ulps_in31.max().item():.1f} "
          f"(vs {ulps_out31.max().item():.1f} ULPs of output)")

    # Final entry of hf_hidden is AFTER final_layer_norm: compare against TL's
    # ln_final output (w,b applied above). LN divides each position by its own
    # std, so the flat ULP-of-tensor-max criterion misreads this row; use the
    # propagation-aware per-position bound described in the docstring.
    max_abs, scale, ulps, _ = diff_stats(tl_ln_final, hf_hidden[-1])
    x = tl_resids[-1][0]  # final raw residual, (seq, d_model); ~1 ULP off HF's
    sigma = (x.var(dim=-1, unbiased=False) + 1e-5).sqrt()  # per-position LN std
    per_pos_diff = (tl_ln_final[0] - hf_hidden[-1][0]).abs().max(dim=-1).values
    allowed = N_ULPS * ulp * (
        x.abs().max(dim=-1).values / sigma
        + hf_hidden[-1][0].abs().max(dim=-1).values
    )
    ratio = (per_pos_diff / allowed).max().item()
    ok = ratio <= 1.0
    marker = "" if ok else "  <-- diverges"
    print(f"{'post-final-LN (ln_final)':>28} {max_abs:12.4e} {scale:10.3f} "
          f"{max_abs / scale:9.2e} {ulps:6.1f}  {ok}{marker}")
    print(f"{'':>28} (per-position propagation bound: worst position at "
          f"{ratio:.2f}x its allowance)")
    if not ok and first_divergence is None:
        first_divergence = "final LayerNorm"

    raw_vs_normed = (tl_resids[-1] - hf_hidden[-1]).abs().max().item()

    print("\n=== Verdict ===")
    print(f"TL final-layer residual: {final_hook}, shape {tuple(tl_resids[-1].shape)}")
    print(f"HF comparison point for it: hidden_states[-1] is post-final-LN in "
          f"GPT-NeoX (raw-vs-normed max abs diff {raw_vs_normed:.3f}, expected large), "
          f"so the final-depth check is {ln_final_hook} vs hidden_states[-1].")
    if first_divergence is None:
        print(f"PASS: all depths agree within {N_ULPS} {args.dtype} ULPs "
              f"({args.dtype} rounding + summation-order differences).")
    else:
        print(f"FAIL: first divergence beyond {N_ULPS} {args.dtype} ULPs "
              f"at: {first_divergence}. "
              "Depths before it agree; see the per-layer table above.")


if __name__ == "__main__":
    main()
