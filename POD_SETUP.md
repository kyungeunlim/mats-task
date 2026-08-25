# Pod restart checklist

After starting or redeploying the pod (US-KS-2, volume `established_copper_caribou`):

1. Copy the new SSH host/port from the pod's Connect panel ("SSH over exposed TCP").
   Update HostName and Port in the `runpod` block of ~/.ssh/config on the laptop.
   Connect with `ssh -A runpod` (the -A is needed for git on the private repo).
   Test: `nvidia-smi`

2. On the pod, restore container-disk tools and set up the shell:
   cd /workspace/mats-task
   ./pod_bootstrap.sh
   source pod_env.sh

3. Verify:
   python -c "import pandas, torch, transformer_lens; print(pandas.__version__, torch.cuda.is_available())"
   Expect: 3.0.5 True

## Notes
- Models cached at /workspace/hf/hub (base, e2e-strong-filter, no-LoRA CB)
- Datasets cached: EleutherAI/wmdp_bio_cloze (1076 cloze_compatible, 197 mcqa_only),
  cais/wmdp wmdp-bio (1273), cais/mmlu all (14042)
- First imports from the volume are slow. Roughly 1-2 min on the A100 pod,
  about 5 min on the L40 pod. Not a hang.
- If a package install fails with "Stale file handle", retry; if it repeats,
  put the venv on container disk with UV_CACHE_DIR=/workspace/uv-cache
- Stop the pod at end of day. Idle cost $0.00/hr; volume is $7/mo.

## If the pod won't resume
A stopped pod is pinned to its original host, and resume fails if that host has no
free GPU. This happened on Aug 25. A100s in KS-2 were unavailable both mornings.

Options in order: retry once or twice, then terminate the pod (leave the
"Also delete attached network volume" checkbox UNCHECKED) and redeploy through the
Deploy form with `established_copper_caribou` selected. Any 40GB+ card works for
one 7B model at a time: L40 ($0.82) and RTX A6000 ($0.53) are the usual fallbacks.
Migration ("Automatically migrate your Pod data") only works if an identical GPU is
free, so it often isn't an option.

