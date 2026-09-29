#!/usr/bin/env bash
set -euo pipefail

ENV_NAMES=(final_01567 final_2 final_3 final_4 final_allinone)
YES=0
DRY_RUN=0
CLEAN_CACHE=0

usage() {
  cat <<'EOF'
Usage: scripts/remove_conda_envs.sh [options]

Removes the Conda environments created for this project:
  final_01567 final_2 final_3 final_4 final_allinone

Options:
  --yes, -y       Do not prompt before removing environments.
  --dry-run       Show what would be removed, but do not remove anything.
  --clean-cache   Run `conda clean --all --yes` after removing envs.
  --help, -h      Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y) YES=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --clean-cache) CLEAN_CACHE=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH." >&2
  exit 1
fi

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "$1"
}

existing_envs=()
missing_envs=()
for name in "${ENV_NAMES[@]}"; do
  if env_exists "$name"; then
    existing_envs+=("$name")
  else
    missing_envs+=("$name")
  fi
done

if [[ "${#existing_envs[@]}" -eq 0 ]]; then
  echo "[cleanup] No project Conda environments found. Nothing to remove."
  if [[ "$CLEAN_CACHE" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
    echo "[cleanup] Cleaning Conda package caches..."
    conda clean --all --yes
  fi
  exit 0
fi

echo "[cleanup] Project environments found:"
printf '  - %s\n' "${existing_envs[@]}"

if [[ "${#missing_envs[@]}" -gt 0 ]]; then
  echo "[cleanup] Already absent:"
  printf '  - %s\n' "${missing_envs[@]}"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[cleanup] Dry run only; no environments removed."
  exit 0
fi

if [[ "$YES" -ne 1 ]]; then
  printf '[cleanup] Remove these environments? Type "yes" to continue: '
  read -r answer
  if [[ "$answer" != "yes" ]]; then
    echo "[cleanup] Aborted."
    exit 1
  fi
fi

for name in "${existing_envs[@]}"; do
  echo "[cleanup] Removing $name..."
  conda env remove -n "$name" --yes
done

if [[ "$CLEAN_CACHE" -eq 1 ]]; then
  echo "[cleanup] Cleaning Conda package caches..."
  conda clean --all --yes
fi

echo "[cleanup] Done."
