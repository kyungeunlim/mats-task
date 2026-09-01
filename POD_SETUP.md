# Pod restart checklist

Assume every morning is a redeploy, not a resume. See "If the pod won't resume" below.

After deploying the pod (US-KS-2, volume `established_copper_caribou`):

1. Copy the new SSH host/port from the pod's Connect panel ("SSH over exposed TCP").
   Update HostName and Port in the `runpod` block of ~/.ssh/config on the laptop.
   Connect with `ssh -A runpod` (the -A is needed for git on the private repo).
   Test: `nvidia-smi`

2. On the pod:
   cd /workspace/mats-task
   git pull
   ./pod_bootstrap.sh     # 5-30 min: installs uv, Claude Code, recreates the venv on container disk
   source ~/.bashrc       # or open a new shell

   Start bootstrap first, then do laptop-side work while it runs.

3. Verify:
   python -c "import pandas, torch, transformer_lens; print(pandas.__version__, torch.cuda.is_available())"
   Expect: 3.0.5 True

## Notes
- Models cached at /workspace/hf/hub (base, e2e-strong-filter, no-LoRA CB)
- Datasets cached: EleutherAI/wmdp_bio_cloze (1076 cloze_compatible, 197 mcqa_only),
  cais/wmdp wmdp-bio (1273), cais/mmlu all (14042)
- Python env lives at /root/venv on container disk, recreated by bootstrap from
  requirements.txt on every pod. The volume is far too slow for many small files
  (imports, harness task index, git status). Volume is for bulk data only: models,
  datasets, cached activations, results. Do not put a venv on the volume.
- Model weight loads from the volume take a few minutes. Bulk reads of large files
  are fine, small-file access is not.
- HF_HUB_OFFLINE=1 is set by pod_env.sh since models are cached. Unset it if you
  need to download something new.
- Stop the pod at end of day. Idle cost $0.00/hr; volume is $7/mo.

## If the pod won't resume
A stopped pod is pinned to its original host, and resume fails if that host has no
free GPU. Resume failed on Aug 25 and Aug 26, every attempt. Treat resume as
unreliable in KS-2 and go straight to redeploy.

Terminate the pod (leave the "Also delete attached network volume" checkbox
UNCHECKED) and redeploy through the Deploy form with `established_copper_caribou`
selected in the network volume dropdown. Check the Region line reads US-KS-2 before
deploying. Any 40GB+ card works for one 7B model at a time: A100 SXM ($1.59) if
free, otherwise L40 ($0.82) or RTX A6000 ($0.53). Migration ("Automatically migrate
your Pod data") only works if an identical GPU is free, so it usually isn't an
option.

Redeploy can land a better card than resume would have (Aug 26: L40 pod wouldn't
resume, redeploy got an A100).

