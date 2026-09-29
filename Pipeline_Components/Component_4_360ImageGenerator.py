import argparse
import json
import os
import random
import sys

# Prevent OpenBLAS warning and potential hangs
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import torch

from import_source import DEFAULT_WORKSPACE, check_triton_version, PanoramaT2I

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_COMPONENT_DIR, ".."))
_DEFAULT_SDXL_MODEL = os.path.join(_PROJECT_ROOT, "big_files", "sdxl_model")
check_triton_version()


def _parse_cli():
    p = argparse.ArgumentParser(
        description="Generate 360° keyframe images for a workspace (interactive range unless --all or --start/--end).",
    )
    p.add_argument(
        "workspace",
        nargs="?",
        default=DEFAULT_WORKSPACE,
        help=f"Run folder containing args.json / raw_residuals.pt (default: {DEFAULT_WORKSPACE})",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Render all keyframes (0 .. N-1), non-interactive.",
    )
    p.add_argument(
        "--start-from-keyframe",
        type=int,
        default=None,
        metavar="I",
        help="First keyframe index (inclusive). Must be used with --end-at-keyframe.",
    )
    p.add_argument(
        "--end-at-keyframe",
        type=int,
        default=None,
        metavar="J",
        help="Last keyframe index (inclusive). Must be used with --start-from-keyframe.",
    )
    return p.parse_args()


def _prompt_keyframe_range(max_keyframes: int) -> tuple[int, int]:
    """Blocks on stdin until the user enters a valid inclusive range."""
    print(f"[Component 4] Total keyframes in song: {max_keyframes}")
    print(
        "[Component 4] Waiting for keyboard input (not frozen). "
        "Or re-run with --all or --start-from-keyframe / --end-at-keyframe to skip prompts."
    )
    last = max_keyframes - 1
    while True:
        try:
            start_input = input(f"[Component 4] Enter start_from_keyframe (0 to {last}): ").strip()
            start_from_keyframe = int(start_input)
            if start_from_keyframe < 0 or start_from_keyframe >= max_keyframes:
                print(f"[Component 4] Invalid input. Please enter a number between 0 and {last}.")
                continue
            break
        except ValueError:
            print("[Component 4] Invalid input. Please enter a valid integer.")

    while True:
        try:
            end_input = input(
                f"[Component 4] Enter end_at_keyframe ({start_from_keyframe} to {last}, inclusive): "
            ).strip()
            end_at_keyframe = int(end_input)
            if end_at_keyframe < start_from_keyframe or end_at_keyframe > last:
                print(
                    f"[Component 4] Invalid input. Please enter a number between "
                    f"{start_from_keyframe} and {last}."
                )
                continue
            break
        except ValueError:
            print("[Component 4] Invalid input. Please enter a valid integer.")
    return start_from_keyframe, end_at_keyframe


def _sorted_section_boundaries(full_sections: list) -> list[float]:
    """Sorted unique section start times (boundaries between full_sections)."""
    if not full_sections:
        return []
    starts = []
    for s in full_sections:
        if isinstance(s, dict) and "start" in s:
            starts.append(float(s["start"]))
    return sorted(set(starts))


def _crosses_any_section_boundary(t0: float, t1: float, boundaries: list[float]) -> bool:
    """True if moving from t0 to t1 (forward) crosses at least one boundary strictly inside (t0, t1]."""
    if t1 <= t0 or not boundaries:
        return False
    for b in boundaries:
        if t0 < b <= t1:
            return True
    return False


def _keyframe_start_sec(kf: int, args: dict, gen_input: dict) -> float:
    db = gen_input.get("downbeat_frame") if gen_input else None
    if isinstance(db, list) and 0 <= kf < len(db) and isinstance(db[kf], dict):
        if "start_sec" in db[kf]:
            return float(db[kf]["start_sec"])
    return float(args.get("start_sec", 0.0)) + float(args.get("four_bar_sec", 1.0)) * kf


def _seeds_per_keyframe_crossing_sections(
    start_from_keyframe: int,
    end_at_keyframe: int,
    base_seed: int,
    args: dict,
    gen_input: dict,
) -> list[int]:
    """One seed per keyframe; randomize once when the timeline crosses any full_sections start."""
    boundaries = _sorted_section_boundaries(args.get("full_sections") or [])
    n = end_at_keyframe - start_from_keyframe + 1
    if not boundaries:
        return [base_seed] * n

    seeds_out: list[int] = []
    current_seed = base_seed
    for i in range(n):
        kf = start_from_keyframe + i
        if kf > 0:
            t0 = _keyframe_start_sec(kf - 1, args, gen_input)
            t1 = _keyframe_start_sec(kf, args, gen_input)
            if _crosses_any_section_boundary(t0, t1, boundaries):
                current_seed = random.randint(0, 2**31 - 1)
                print(
                    f"[Component 4] Crossed full_sections boundary between keyframe {kf - 1} and {kf}; "
                    f"new seed={current_seed}"
                )
        seeds_out.append(current_seed)
    return seeds_out


def main():
    cli = _parse_cli()
    workspace = cli.workspace

    with open(os.path.join(workspace, "args.json"), "r") as f:
        args = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_residuals = torch.load(os.path.join(workspace, "raw_residuals.pt"), map_location=device)
    va_tuples = torch.load(os.path.join(workspace, "va_tuples.pt"), map_location=device)
    
    # Load gen_input.json if it exists (created by Component 3)
    gen_input_path = os.path.join(workspace, "gen_input.json")
    if os.path.exists(gen_input_path):
        with open(gen_input_path, "r") as f:
            gen_input = json.load(f)
    else:
        gen_input = {}
    
    max_keyframes = raw_residuals.shape[0]
    stored_max = args.get("max_keyframes")
    if stored_max is not None and stored_max != max_keyframes:
        print(f"[Component 4] Warning: args.json max_keyframes={stored_max} differs from raw_residuals length={max_keyframes}. Using actual keyframe count.")
    
    last = max_keyframes - 1
    if cli.all:
        start_from_keyframe, end_at_keyframe = 0, last
        print(f"[Component 4] --all: using keyframes 0 .. {last} ({max_keyframes} frames).")
    elif cli.start_from_keyframe is not None or cli.end_at_keyframe is not None:
        if cli.start_from_keyframe is None or cli.end_at_keyframe is None:
            print(
                "[Component 4] ERROR: pass both --start-from-keyframe and --end-at-keyframe, or use --all.",
                file=sys.stderr,
            )
            sys.exit(2)
        start_from_keyframe = cli.start_from_keyframe
        end_at_keyframe = cli.end_at_keyframe
        if start_from_keyframe < 0 or start_from_keyframe >= max_keyframes:
            print(f"[Component 4] ERROR: start must be in [0, {last}].", file=sys.stderr)
            sys.exit(2)
        if end_at_keyframe < start_from_keyframe or end_at_keyframe > last:
            print(
                f"[Component 4] ERROR: end must be in [{start_from_keyframe}, {last}] (inclusive).",
                file=sys.stderr,
            )
            sys.exit(2)
        print(
            f"[Component 4] Using keyframes {start_from_keyframe} .. {end_at_keyframe} "
            f"({end_at_keyframe - start_from_keyframe + 1} frames)."
        )
    else:
        start_from_keyframe, end_at_keyframe = _prompt_keyframe_range(max_keyframes)
    
    # Store the keyframe range in args
    args["start_from_keyframe"] = start_from_keyframe
    args["end_at_keyframe"] = end_at_keyframe
    with open(os.path.join(workspace, "args.json"), "w") as f:
        json.dump(args, f, indent=4)
    
    # Filter residuals and va_tuples by the range
    filtered_raw_residuals = raw_residuals[start_from_keyframe:end_at_keyframe + 1]
    filtered_va_tuples = va_tuples[start_from_keyframe:end_at_keyframe + 1]
    
    sdxl_path = args.get("sdxl_path") or _DEFAULT_SDXL_MODEL
    if not os.path.exists(sdxl_path) and os.path.exists(_DEFAULT_SDXL_MODEL):
        print(f"[Component 4] SDXL path not found at {sdxl_path}; using {_DEFAULT_SDXL_MODEL}")
        sdxl_path = _DEFAULT_SDXL_MODEL
    pano = PanoramaT2I(sdxl_path=sdxl_path, device=str(device))
    seeds = _seeds_per_keyframe_crossing_sections(
        start_from_keyframe,
        end_at_keyframe,
        int(args["seed"]),
        args,
        gen_input,
    )

    print(f"[Component 4] Rendering {filtered_raw_residuals.shape[0]} keyframes (from {start_from_keyframe} to {end_at_keyframe})...")
    pano.generate_batch(
        residuals=filtered_raw_residuals, prompt=args["prompt"], negative_prompt=args["negative_prompt"],
        output_dir=workspace, annotate=False,
        va_tuples=filtered_va_tuples, seeds=seeds, prefix="keyframe"
    )
    
    # Update gen_input.json to only include the filtered keyframe range
    # First, check if gen_input has valid data; if not, regenerate it
    if not gen_input or not gen_input.get("downbeat_frame") or not gen_input.get("dynamic_frames"):
        print(f"[Component 4] WARNING: gen_input.json is missing or incomplete. Regenerating metadata...")
        
        # Rebuild downbeat_frame and frame arrays from filtered va_tuples
        interval_sec = args.get("four_bar_sec", 1.0)
        frame_interval = int(round(interval_sec * args.get("fps", 24)))
        
        downbeat_frame = []
        for i in range(filtered_va_tuples.shape[0]):
            downbeat_frame.append({
                "frame": int(frame_interval * i),
                "start_sec": args.get("start_sec", 0.0) + (interval_sec * i),
                "valence": float(filtered_va_tuples[i, 0].item()),
                "arousal": float(filtered_va_tuples[i, 1].item())
            })
        
        dynamic_frames = []
        transition_frames = []
        for i in range(filtered_va_tuples.shape[0] - 1):
            dyn = int(round(frame_interval * 0.75))
            dynamic_frames.append(dyn)
            transition_frames.append(frame_interval - dyn)
        
        gen_input["downbeat_frame"] = downbeat_frame
        gen_input["dynamic_frames"] = dynamic_frames
        gen_input["transition_frames"] = transition_frames
        gen_input["fps"] = args.get("fps", 24)
        gen_input["prompt"] = args.get("prompt", "")
        print(f"[Component 4] Regenerated: {len(downbeat_frame)} downbeat entries, {len(dynamic_frames)} dynamic_frames, {len(transition_frames)} transition_frames.")
    else:
        # Filter existing gen_input data by keyframe range
        if "downbeat_frame" in gen_input:
            filtered_downbeat_frame = gen_input["downbeat_frame"][start_from_keyframe:end_at_keyframe + 1]
            gen_input["downbeat_frame"] = filtered_downbeat_frame
            print(f"[Component 4] Filtered downbeat_frame from {len(gen_input.get('downbeat_frame', []))} to {len(filtered_downbeat_frame)} entries.")
            
            # Update start_sec to the start_sec of the first filtered keyframe
            if filtered_downbeat_frame:
                gen_input["start_sec"] = filtered_downbeat_frame[0].get("start_sec", gen_input.get("start_sec"))
        
        # Adjust dynamic_frames and transition_frames accordingly
        if "dynamic_frames" in gen_input:
            original_len = len(gen_input["dynamic_frames"])
            gen_input["dynamic_frames"] = gen_input["dynamic_frames"][start_from_keyframe:end_at_keyframe]
            print(f"[Component 4] Filtered dynamic_frames from {original_len} to {len(gen_input['dynamic_frames'])} entries.")
        
        if "transition_frames" in gen_input:
            original_len = len(gen_input["transition_frames"])
            gen_input["transition_frames"] = gen_input["transition_frames"][start_from_keyframe:end_at_keyframe]
            print(f"[Component 4] Filtered transition_frames from {original_len} to {len(gen_input['transition_frames'])} entries.")
    
    # Save the filtered/regenerated gen_input
    with open(gen_input_path, "w") as f:
        json.dump(gen_input, f, indent=4)
    print(f"[Component 4] Updated gen_input.json.")
    
    print("[Component 4] Done.")
if __name__ == "__main__":
    main()