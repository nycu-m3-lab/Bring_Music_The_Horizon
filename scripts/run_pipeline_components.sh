#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPONENT_DIR="$ROOT_DIR/Pipeline_Components"
WORKSPACE=""
FORCE=0
NORMALIZE=1

usage() {
  cat <<'EOF'
Usage: scripts/run_pipeline_components.sh [options] [WORKSPACE]

Runs Components 1-7 in the required Conda environments. The runner is
checkpoint-aware: if an expected output already exists, that component is
skipped so interrupted runs can be resumed safely.

Options:
  --force         Re-run every component even when outputs already exist.
  --no-normalize  Skip Component 5.5 and concatenate clips from WORKSPACE.
  -h, --help      Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1 ;;
    --no-normalize) NORMALIZE=0 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    *)
      if [[ -n "$WORKSPACE" ]]; then
        echo "Unexpected extra argument: $1" >&2
        usage
        exit 2
      fi
      WORKSPACE="$1"
      ;;
  esac
  shift
done

WORKSPACE="${WORKSPACE:-$COMPONENT_DIR/workspace}"

run_component() {
  local env_name="$1"
  local script_name="$2"
  shift 2
  echo "[pipeline] ($env_name) $script_name $*"
  conda run --no-capture-output -n "$env_name" python "$COMPONENT_DIR/$script_name" "$@"
}

run_if_needed() {
  local label="$1"
  local check_func="$2"
  local env_name="$3"
  local script_name="$4"
  shift 4

  if [[ "$FORCE" -eq 0 ]] && "$check_func"; then
    echo "[pipeline] Skip $label; checkpoint already exists."
    return 0
  fi

  run_component "$env_name" "$script_name" "$@"
}

json_has_fields() {
  local json_path="$1"
  shift
  python - "$json_path" "$@" <<'PYCHECK'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
fields = sys.argv[2:]
if not path.exists():
    raise SystemExit(1)
try:
    data = json.loads(path.read_text())
except Exception:
    raise SystemExit(1)
for field in fields:
    if field not in data or data[field] in (None, [], ""):
        raise SystemExit(1)
PYCHECK
}

has_structure() {
  json_has_fields "$WORKSPACE/args.json" full_downbeats bpm four_bar_sec hz max_keyframes
}

has_va() {
  [[ -s "$WORKSPACE/va_tuples.pt" ]]
}

has_residuals() {
  [[ -s "$WORKSPACE/raw_residuals.pt" && -s "$WORKSPACE/gen_input.json" ]]
}

keyframes_complete() {
  python - "$WORKSPACE" <<'PYCHECK'
import glob
import json
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
args_path = workspace / "args.json"
if not args_path.exists():
    raise SystemExit(1)
args = json.loads(args_path.read_text())
expected = int(args.get("max_keyframes") or 0)
if expected <= 0:
    raise SystemExit(1)
count = len(glob.glob(str(workspace / "keyframe_*.png")))
raise SystemExit(0 if count >= expected else 1)
PYCHECK
}

segments_complete() {
  python - "$WORKSPACE" <<'PYCHECK'
import json
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
gen_path = workspace / "gen_input.json"
if not gen_path.exists():
    raise SystemExit(1)
gen = json.loads(gen_path.read_text())
downbeats = gen.get("downbeat_frame") or []
transition_frames = gen.get("transition_frames") or []
num_segments = max(len(downbeats) - 1, 0)
if num_segments <= 0:
    raise SystemExit(1)
for i in range(num_segments):
    i2v = workspace / f"seg{i:03d}_i2v.mp4"
    if not i2v.exists() or i2v.stat().st_size <= 1024:
        raise SystemExit(1)
    needs_transition = i < len(transition_frames) and int(transition_frames[i] or 0) > 0
    flf = workspace / f"seg{i:03d}_flf2v.mp4"
    if needs_transition and (not flf.exists() or flf.stat().st_size <= 1024):
        raise SystemExit(1)
raise SystemExit(0)
PYCHECK
}

normalized_complete() {
  python - "$WORKSPACE" <<'PYCHECK'
import glob
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
out_dir = workspace / "normalized_16x9"
if not (out_dir / "args.json").exists() or not (out_dir / "gen_input.json").exists():
    raise SystemExit(1)
sources = sorted(Path(p) for p in glob.glob(str(workspace / "seg*.mp4")))
if not sources:
    raise SystemExit(1)
for src in sources:
    dst = out_dir / src.name
    if not dst.exists() or dst.stat().st_size <= 1024:
        raise SystemExit(1)
raise SystemExit(0)
PYCHECK
}

has_silent_video() {
  [[ -s "$PIPE_WORKSPACE/final_silent_video.mp4" || -s "$PIPE_WORKSPACE/final_silent_video_partial.mp4" ]]
}

has_music_video() {
  [[ -s "$PIPE_WORKSPACE/final_music_video.mp4" || -s "$PIPE_WORKSPACE/final_music_video_partial.mp4" ]]
}

if [[ ! -f "$WORKSPACE/args.json" ]]; then
  cat >&2 <<EOF
Workspace is missing args.json: $WORKSPACE
Run Component_0_args_saver.py first, for example:
  conda run --no-capture-output -n final_01567 python "$COMPONENT_DIR/Component_0_args_saver.py" AUDIO_PATH PROMPT --workspace "$WORKSPACE"
EOF
  exit 1
fi

export ALLIN1_CONDA_ENV="${ALLIN1_CONDA_ENV:-final_allinone}"
export FFMPEG="${FFMPEG:-$(command -v ffmpeg || true)}"
export FFPROBE="${FFPROBE:-$(command -v ffprobe || true)}"

run_if_needed "Component 1" has_structure final_01567 Component_1_allinone.py "$WORKSPACE"
run_if_needed "Component 2" has_va final_2 Component_2_VAExtractor.py "$WORKSPACE"
run_if_needed "Component 3" has_residuals final_3 Component_3_Residual_Calculator.py "$WORKSPACE"
run_if_needed "Component 4" keyframes_complete final_4 Component_4_360ImageGenerator.py "$WORKSPACE" --all
run_if_needed "Component 5" segments_complete final_01567 Component_5_360DynamicAndTransitionGenerator.py "$WORKSPACE"

PIPE_WORKSPACE="$WORKSPACE"
if [[ "$NORMALIZE" -eq 1 && -f "$COMPONENT_DIR/Component_5.5_NormalizeClips16x9.py" ]]; then
  run_if_needed "Component 5.5" normalized_complete final_01567 Component_5.5_NormalizeClips16x9.py "$WORKSPACE"
  PIPE_WORKSPACE="$WORKSPACE/normalized_16x9"
fi

run_if_needed "Component 6" has_silent_video final_01567 Component_6_Concatenator.py "$PIPE_WORKSPACE"
run_if_needed "Component 7" has_music_video final_01567 Component_7_MusicAdder.py "$PIPE_WORKSPACE"

echo "[pipeline] Done. Active output workspace: $PIPE_WORKSPACE"
