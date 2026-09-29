#!/usr/bin/env python3
"""Temporary helper to normalize generated segment clips to 16:9.

This center-crops wide panorama I2V clips, scales every segment to 1280x720,
and can concatenate the normalized copies. Originals are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import shutil
from pathlib import Path


FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "/mnt/M3_Lab/Chiikawa/ffmpeg/ffmpeg"
FFPROBE = os.environ.get("FFPROBE") or shutil.which("ffprobe") or "/mnt/M3_Lab/Chiikawa/ffmpeg/ffprobe"
DEFAULT_WORKSPACE = Path(__file__).resolve().parent / "nirvana_1st_run"


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd))
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


def center_crop_16x9(width: int, height: int) -> tuple[int, int, int, int]:
    target_ratio = 16 / 9
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


def normalize_clip(src: Path, dst: Path, fps: int, width: int, height: int) -> None:
    src_w, src_h = probe_size(src)
    crop_w, crop_h, crop_x, crop_y = center_crop_16x9(src_w, src_h)
    vf = (
        f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
        f"scale={width}:{height},setsar=1,fps={fps},format=yuv420p"
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            FFMPEG,
            "-y",
            "-i",
            str(src),
            "-vf",
            vf,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(dst),
        ]
    )


def concat_clips(clips: list[Path], output: Path, fps: int) -> None:
    list_file = output.parent / "_normalized_concat_list.txt"
    with open(list_file, "w") as f:
        for clip in clips:
            f.write(f"file '{clip.resolve()}'\n")

    run(
        [
            FFMPEG,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            "-r",
            str(fps),
            str(output),
        ]
    )
    list_file.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize segXXX_i2v.mp4 and segXXX_flf2v.mp4 clips to 1280x720."
    )
    parser.add_argument("workspace", nargs="?", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--concat", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    out_dir = (args.out_dir or workspace / "normalized_16x9").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    normalized: list[Path] = []
    i2v_files = sorted(workspace.glob("seg*_i2v.mp4"))
    if not i2v_files:
        raise SystemExit(f"No seg*_i2v.mp4 files found in {workspace}")

    for i2v in i2v_files:
        seg = i2v.name[:6]
        for src in (i2v, workspace / f"{seg}_flf2v.mp4"):
            if not src.exists():
                continue
            dst = out_dir / src.name
            print(f"[normalize] {src.name} -> {dst.relative_to(workspace)}")
            normalize_clip(src, dst, args.fps, args.width, args.height)
            normalized.append(dst)

    if args.concat:
        output = out_dir / "final_silent_video_16x9.mp4"
        concat_clips(normalized, output, args.fps)
        print(f"[done] {output}")
    else:
        print(f"[done] normalized clips written to {out_dir}")


if __name__ == "__main__":
    main()
