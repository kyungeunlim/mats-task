"""Smoke test: load GPT-2-small on CPU, run one forward pass, plot residual
stream norm at each layer.

Run as: python smoke_test.py
Outputs: results/smoke_test.png (mean over all positions)
         results/smoke_test_pos0.png (mean incl. vs excl. position 0)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; we only save to disk
import matplotlib.pyplot as plt
import torch
from transformer_lens import HookedTransformer

PROMPT = "The Eiffel Tower is in the city of"
OUT_PATH = Path("results/smoke_test.png")
OUT_PATH_POS0 = Path("results/smoke_test_pos0.png")


def main() -> None:
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")

    tokens = model.to_tokens(PROMPT)  # shape: [1, seq_len]
    print(f"Prompt: {PROMPT!r}")
    print(f"Token shape: {tuple(tokens.shape)}")

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
    print(f"Logits shape: {tuple(logits.shape)}")

    # resid_post[layer]: [1, seq_len, d_model]. Per-position L2 norm over
    # d_model gives [seq_len]; from that we take the mean over all positions,
    # the mean excluding position 0, and position 0's norm alone.
    norms_all = []
    norms_no_pos0 = []
    norms_pos0 = []
    for layer in range(model.cfg.n_layers):
        per_pos = cache["resid_post", layer].norm(dim=-1)[0]  # [seq_len]
        norms_all.append(per_pos.mean().item())
        norms_no_pos0.append(per_pos[1:].mean().item())
        norms_pos0.append(per_pos[0].item())
        print(
            f"layer {layer:2d}: mean={norms_all[-1]:8.3f}  "
            f"mean_excl_pos0={norms_no_pos0[-1]:8.3f}  "
            f"pos0={norms_pos0[-1]:8.3f}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    layers = range(model.cfg.n_layers)

    # Original single-curve plot (mean over all positions), kept for continuity.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(layers, norms_all, marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean L2 norm of resid_post")
    ax.set_title(f"GPT-2-small residual stream norm\nprompt: {PROMPT!r}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Saved plot to {OUT_PATH}")

    # Comparison plot: mean including vs excluding position 0.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(layers, norms_all, marker="o", label="mean (all positions)")
    ax.plot(layers, norms_no_pos0, marker="s", label="mean (excl. position 0)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean L2 norm of resid_post")
    ax.set_title(
        f"GPT-2-small residual stream norm, with/without position 0\n"
        f"prompt: {PROMPT!r}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH_POS0, dpi=150)
    print(f"Saved plot to {OUT_PATH_POS0}")


if __name__ == "__main__":
    main()
