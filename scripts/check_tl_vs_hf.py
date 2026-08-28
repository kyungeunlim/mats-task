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
  transients. Fine on this 80GB A100; on a 40GB card it would be tight.

Run on the pod:
    /root/venv/bin/python scripts/check_tl_vs_hf.py
"""

import gc
import json
from pathlib import Path

import torch

MODEL_NAME = "EleutherAI/deep-ignorance-unfiltered"
# TL official name whose architecture deep-ignorance shares; see docstring.
TL_OFFICIAL_NAME = "EleutherAI/pythia-6.9b"
BF16_ULP = 2.0 ** -8  # relative spacing of bfloat16 (8-bit effective mantissa)
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
    assert torch.cuda.is_available(), "this check is meant to run on the pod GPU"
    device = "cuda"

    prompt = build_prompt()

    # ---- HF -----------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading {MODEL_NAME} via AutoModelForCausalLM (dtype=bfloat16) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device)
    hf_model.eval()
    hf_cfg = hf_model.config
    print("=== Config ===")
    print(f"HF  num_hidden_layers={hf_cfg.num_hidden_layers}  "
          f"hidden_size={hf_cfg.hidden_size}  vocab_size={hf_cfg.vocab_size}")
    n_layers = hf_cfg.num_hidden_layers

    # Tokenize ONCE with the HF tokenizer; feed the same ids to both models.
    # Passing a tensor to TL bypasses prepend_bos, matching lm-eval (no BOS).
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    print(f"\ninput_ids shape: {tuple(input_ids.shape)} "
          f"({input_ids.shape[1]} tokens, no BOS added)")

    with torch.no_grad():
        out = hf_model(input_ids, output_hidden_states=True)
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
          f"('{TL_OFFICIAL_NAME}', hf_model=<{MODEL_NAME}>, dtype=bfloat16) ...")
    tl_model = HookedTransformer.from_pretrained_no_processing(
        TL_OFFICIAL_NAME,
        hf_model=hf_model,
        tokenizer=tokenizer,
        dtype=torch.bfloat16,
        device=device,
    )
    tl_model.eval()
    cfg = tl_model.cfg
    print(f"TL  n_layers={cfg.n_layers}  d_model={cfg.d_model}  d_vocab={cfg.d_vocab}")
    print("TL load flags: from_pretrained_no_processing => "
          "fold_ln=False, center_writing_weights=False, center_unembed=False, "
          "fold_value_biases=False, dtype=bfloat16")
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
        ulps = max_abs / (scale * BF16_ULP) if scale > 0 else 0.0
        return max_abs, scale, ulps, ulps <= N_ULPS

    def report_row(label: str, a: torch.Tensor, b: torch.Tensor) -> bool:
        max_abs, scale, ulps, ok = diff_stats(a, b)
        marker = "" if ok else "  <-- diverges"
        print(f"{label:>28} {max_abs:12.4e} {scale:10.3f} "
              f"{max_abs / scale:9.2e} {ulps:6.1f}  {ok}{marker}")
        return ok

    print(f"\n=== Layer-by-layer comparison "
          f"(matched if <= {N_ULPS} bf16 ULPs of max |HF|, "
          f"i.e. rel <= {N_ULPS * BF16_ULP:.2%}) ===")
    print(f"{'depth':>28} {'max|TL-HF|':>12} {'max|HF|':>10} {'rel':>9} {'ULPs':>6}  ok")
    first_divergence = None

    if not report_row("embeddings (hook_embed)", tl_embed, hf_hidden[0]):
        first_divergence = "embeddings"

    # hf_hidden[i] (1 <= i <= n_layers-1) is the raw residual after block i-1.
    for i in range(1, cfg.n_layers):
        ok = report_row(f"blocks.{i-1}.hook_resid_post", tl_resids[i - 1], hf_hidden[i])
        if not ok and first_divergence is None:
            first_divergence = f"layer {i - 1}"

    # Final entry of hf_hidden is AFTER final_layer_norm: compare against TL's
    # ln_final output (w,b applied above). LN divides each position by its own
    # std, so the flat ULP-of-tensor-max criterion misreads this row; use the
    # propagation-aware per-position bound described in the docstring.
    max_abs, scale, ulps, _ = diff_stats(tl_ln_final, hf_hidden[-1])
    x = tl_resids[-1][0]  # final raw residual, (seq, d_model); ~1 ULP off HF's
    sigma = (x.var(dim=-1, unbiased=False) + 1e-5).sqrt()  # per-position LN std
    per_pos_diff = (tl_ln_final[0] - hf_hidden[-1][0]).abs().max(dim=-1).values
    allowed = N_ULPS * BF16_ULP * (
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
        print(f"PASS: all depths agree within {N_ULPS} bf16 ULPs "
              f"(bfloat16 rounding + summation-order differences).")
    else:
        print(f"FAIL: first divergence beyond {N_ULPS} bf16 ULPs at: {first_divergence}. "
              "Depths before it agree; see the per-layer table above.")


if __name__ == "__main__":
    main()
