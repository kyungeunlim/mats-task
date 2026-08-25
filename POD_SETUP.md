# Pod restart checklist

After starting or redeploying the pod (US-KS-2, A100 SXM, volume `established_copper_caribou`):

1. Copy the new SSH host/port from the pod's Connect panel ("SSH over exposed TCP").
   Update HostName and Port in the `runpod` block of ~/.ssh/config on the laptop.
   Test: `ssh runpod nvidia-smi`

2. On the pod, restore container-disk tools and set up the shell:

   cd /workspace/mats-task
   ./pod_bootstrap.sh
   source pod_env.sh

3. Verify:
   python -c "import pandas, torch, transformer_lens; print(pandas.__version__, torch.cuda.is_available())"
   Expect: 3.0.5 True

## Notes
- Models cached at /workspace/hf/hub (base, e2e-strong-filter, no-LoRA CB)
- Datasets cached: cais/wmdp wmdp-bio (1273), cais/mmlu all (14042)
- First imports from the volume are slow (30-90s). Not a hang.
- If a package install fails with "Stale file handle", retry; if it repeats,
  put the venv on container disk with UV_CACHE_DIR=/workspace/uv-cache
- Use `ssh -A runpod` when git needs auth (private repo)
- Stop the pod at end of day. Idle cost $0.00/hr; volume is $7/mo.
