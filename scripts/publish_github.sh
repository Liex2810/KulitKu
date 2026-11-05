#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
repo="${1:-Liex2810/kulitku}"
command -v gh >/dev/null || { echo 'Install https://cli.github.com'; exit 1; }
gh auth status
if [ ! -d .git ]; then git init -b main; fi
git add .
if ! git diff --cached --quiet; then git commit -m 'Initialize KulitKu monorepo'; fi
gh repo create "$repo" --private --source . --remote origin --push
