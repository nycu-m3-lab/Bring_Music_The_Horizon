#!/usr/bin/env python3
"""Component 5.5: normalize generated I2V/transition clips to 16:9.

This component is intended to run after Component 5 and before Component 6.
It writes a separate normalized workspace so the original Fal outputs remain
available for inspection or reprocessing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from import_source import DEFAULT_WORKSPACE


FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "/mnt/M3_Lab/Chiikawa/ffmpeg/ffmpeg"
FFPROBE = os.environ.get("FFPROBE") or shutil.which("ffprobe") or "/mnt/M3_Lab/Chiikawa/ffmpeg/ffprobe"


def run(cmd: list[str]) -> None:
    print("[Component 5.5]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def probe_size(path: Path) -> tuple[int, int]:
    out = subprocess.check_output(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    stream = json.loads(out)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def center_crop_to_ratio(width: int, height: int, target_ratio: float) -> tuple[int, int, int, int]:
    ratio = width / height

    if ratio > target_ratio:
        crop_h = height
        crop_w = int(crop_h * target_ratio)
        crop_w -= crop_w % 2
    else:
        crop_w = width
        crop_h = int(crop_w / target_ratio)
        crop_h -= crop_h % 2

    crop_x = max((width - crop_w) // 2, 0)
    crop_y = max((height - crop_h) // 2, 0)
    crop_x -= crop_x % 2
    crop_y -= crop_y % 2
    return crop_w, crop_h, crop_x, crop_y


def normalize_clip(src: Path, dst: Path, fps: int, width: int, height: int, crf: int, keep_audio: bool = False) -> None:
    src_w, src_h = probe_size(src)
    crop_w, crop_h, crop_x, crop_y = center_crop_to_ratio(src_w, src_h, width / height)
    vf = (
        f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
        f"scale={width}:{height},setsar=1,fps={fps},format=yuv420p"
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[Component 5.5] {src.name}: {src_w}x{src_h} -> "
        f"crop {crop_w}x{crop_h}+{crop_x}+{crop_y} -> {width}x{height}"
    )
    cmd = [
        FFMPEG,
        "-y",
        "-i",
        str(src),
        "-map",
        "0:v:0",
    ]
    if keep_audio:
        cmd.extend(["-map", "0:a?"])
    cmd.extend(
        [
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
        ]
    )
    if keep_audio:
        cmd.extend(["-c:a", "copy", "-shortest"])
    else:
        cmd.append("-an")
    cmd.extend(["-movflags", "+faststart", str(dst)])
    run(cmd)


def copy_pipeline_metadata(workspace: Path, out_dir: Path) -> None:
    for name in ("args.json", "gen_input.json"):
        src = workspace / name
        if not src.exists():
            raise FileNotFoundError(f"Required metadata file missing: {src}")
        shutil.copy2(src, out_dir / name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize Component 5 segment videos to one 16:9 resolution before Component 6."
    )
    parser.add_argument("workspace", nargs="?", default=DEFAULT_WORKSPACE)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--crf", type=int, default=18)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else workspace / "normalized_16x9"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_input_path = workspace / "gen_input.json"
    if not gen_input_path.exists():
        raise FileNotFoundError(f"gen_input.json not found: {gen_input_path}")
    with open(gen_input_path, "r") as f:
        gen_input = json.load(f)

    fps = args.fps or int(gen_input["fps"])
    copy_pipeline_metadata(workspace, out_dir)

    i2v_files = sorted(workspace.glob("seg*_i2v.mp4"))
    final_candidates = [
        workspace / "final_silent_video.mp4",
        workspace / "final_silent_video_partial.mp4",
        workspace / "final_music_video.mp4",
        workspace / "final_music_video_partial.mp4",
    ]
    existing_final_videos = [p for p in final_candidates if p.exists()]

    if not i2v_files and not existing_final_videos:
        raise SystemExit(f"No seg*_i2v.mp4 or final video files found in {workspace}")

    normalized_count = 0
    for i2v in i2v_files:
        seg = i2v.name[:6]
        for src in (i2v, workspace / f"{seg}_flf2v.mp4"):
            if not src.exists():
                continue
            dst = out_dir / src.name
            if src.resolve() == dst.resolve():
                print(f"[Component 5.5] Skipping {src.name}; source and destination are the same.")
                continue
            normalize_clip(src, dst, fps, args.width, args.height, args.crf)
            normalized_count += 1

    for src in existing_final_videos:
        dst = out_dir / src.name
        if src.resolve() == dst.resolve():
            print(f"[Component 5.5] Skipping {src.name}; source and destination are the same.")
            continue
        keep_audio = src.name.startswith("final_music_video")
        normalize_clip(src, dst, fps, args.width, args.height, args.crf, keep_audio=keep_audio)
        normalized_count += 1

    print(f"[Component 5.5] Normalized {normalized_count} clips/videos -> {out_dir}")
    print(f"[Component 5.5] Next:")
    print(f"  python Component_6_Concatenator.py {out_dir}")
    print(f"  python Component_7_MusicAdder.py {out_dir}")


if __name__ == "__main__":
    main()
