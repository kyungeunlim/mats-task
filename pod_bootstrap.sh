#!/usr/bin/env bash
set -e

# Container-disk tools, lost on every pod stop
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://claude.ai/install.sh | bash

# Claude Code auth state lives on the volume
ln -sfn /workspace/claude-config ~/.claude
ln -sfn /workspace/claude-config.json ~/.claude.json

# Git identity (also container disk)
git config --global user.name "Kyungeun Lim"
git config --global user.email "kyungeunlim@users.noreply.github.com"

echo "Done. Now run: source pod_env.sh"
