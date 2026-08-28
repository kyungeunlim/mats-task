#!/usr/bin/env bash
set -e

# Container-disk tools, lost on every pod stop
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
curl -fsSL https://claude.ai/install.sh | bash

# Claude Code auth state lives on the volume
[ -L ~/.claude ] || rm -rf ~/.claude
[ -L ~/.claude.json ] || rm -f ~/.claude.json
ln -sfn /workspace/claude-config ~/.claude
ln -sfn /workspace/claude-config.json ~/.claude.json

# Git identity and pull behavior (container disk)
git config --global user.name "Kyungeun Lim"
git config --global user.email "kyungeunlim@users.noreply.github.com"
git config --global pull.rebase false

# Python env on container disk. The volume is too slow for many small files.
# Takes 20-30 min per fresh pod. Do laptop work while it runs.
uv venv --python 3.12 /root/venv
uv pip install --python /root/venv/bin/python -r /workspace/mats-task/requirements.txt

# Jupyter kernel from the venv (kernelspec lives on container disk)
/root/venv/bin/python -m ipykernel install --user --name mats-task --display-name "mats-task (venv)"

# Auto-source the env on login
grep -q pod_env.sh ~/.bashrc || echo 'source /workspace/mats-task/pod_env.sh' >> ~/.bashrc

echo "Done. Open a new shell, or run: source ~/.bashrc"
