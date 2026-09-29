#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$ROOT_DIR/envs"

ENV_NAMES=(final_01567 final_2 final_3 final_4 final_allinone)

usage() {
  cat <<'EOF'
Usage: scripts/setup_conda_envs.sh [--yes] [--skip-apt]

Creates/updates the Conda environments required by Pipeline_Components.
The all-in-one environment is installed with the project-specific pip/NATTEN
sequence, not through a Conda allin1 package.

Options:
  --yes       Pass -y to conda env create/update.
  --skip-apt  Do not try to install Ubuntu system packages.
EOF
}

YES_FLAG=""
SKIP_APT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes|-y) YES_FLAG="-y" ;;
    --skip-apt) SKIP_APT=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH." >&2
  exit 1
fi

if [[ "$SKIP_APT" -eq 0 ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    echo "[setup] Installing Ubuntu build/runtime packages with sudo apt-get..."
    sudo apt-get update
    sudo apt-get install -y python3.10-dev build-essential ffmpeg wget
  else
    echo "[setup] apt-get not found; install python3.10-dev, build-essential, ffmpeg, and wget manually if needed."
  fi
fi

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "$1"
}

create_or_update_env() {
  local name="$1"
  local file="$2"
  if env_exists "$name"; then
    echo "[setup] Updating $name from $file"
    conda env update -n "$name" -f "$file" --prune $YES_FLAG
  else
    echo "[setup] Creating $name from $file"
    conda env create -f "$file" $YES_FLAG
  fi
}

create_or_update_env final_01567 "$ENV_DIR/environment-final-01567.yml"
create_or_update_env final_2 "$ENV_DIR/environment-final-2.yml"
create_or_update_env final_3 "$ENV_DIR/environment-final-3.yml"
create_or_update_env final_4 "$ENV_DIR/environment-final-4.yml"
create_or_update_env final_allinone "$ENV_DIR/environment-final-allinone.yml"

echo "[setup] Installing all-in-one stack into final_allinone with the required pip flow..."
conda run --no-capture-output -n final_allinone python -m pip install --upgrade pip
conda run --no-capture-output -n final_allinone python -m pip install \
  torch==2.5.0+cu121 \
  torchvision==0.20.0+cu121 \
  torchaudio==2.5.0+cu121 \
  --extra-index-url https://download.pytorch.org/whl/cu121

NATTEN_WHEEL="$ROOT_DIR/big_files/natten-0.17.5+torch250cu121-cp310-cp310-linux_x86_64.whl"
if [[ ! -f "$NATTEN_WHEEL" ]]; then
  echo "Missing local NATTEN wheel: $NATTEN_WHEEL" >&2
  echo "This repo should include the wheel because the upstream download URL is no longer reliable." >&2
  exit 1
fi
conda run --no-capture-output -n final_allinone python -m pip install "$NATTEN_WHEEL"
conda run --no-capture-output -n final_allinone python -m pip install git+https://github.com/CPJKU/madmom
conda run --no-capture-output -n final_allinone python -m pip install git+https://github.com/govi218/all-in-one.git@gov/update-deprecated

echo "[setup] Done. Component env mapping:"
echo "  0,1,5,5.5,6,7 -> final_01567"
echo "  2                -> final_2"
echo "  3                -> final_3"
echo "  4                -> final_4"
echo "  allin1 worker    -> final_allinone"
