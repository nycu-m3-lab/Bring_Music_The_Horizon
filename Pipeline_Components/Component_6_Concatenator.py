import json
import sys
import os
import shutil
import subprocess

from import_source import DEFAULT_WORKSPACE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
FFPROBE = os.environ.get("FFPROBE") or shutil.which("ffprobe")

if not FFMPEG:
    raise RuntimeError(
        "ffmpeg not found. Install ffmpeg or set the FFMPEG environment variable."
    )

if not FFPROBE:
    raise RuntimeError(
        "ffprobe not found. Install ffmpeg or set the FFPROBE environment variable."
    )

def _probe_video_size(path: str) -> tuple[int, int]:
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
            "csv=p=0",
            path,
        ],
        text=True,
    ).strip()
    width, height = out.split(",")[:2]
    return int(width), int(height)


def _ffmpeg_concat_reencode(clip_paths: list[str], fps: int) -> str:
    """Concatenate clips after normalizing dimensions/timestamps.

    The Fal I2V and FLF2V endpoints may return different frame sizes. Normalize
    one clip at a time first so ffmpeg does not spawn decoders for every input at
    once, then stream-copy concat the normalized temporaries.
    """
    os.makedirs("./results", exist_ok=True)
    temp_dir = os.path.join("./results", "_concat_normalized")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    output_path = os.path.join("./results", "final_output.mp4")
    width, height = _probe_video_size(clip_paths[0])
    normalized_paths = []

    for idx, src in enumerate(clip_paths):
        dst = os.path.join(temp_dir, f"clip_{idx:04d}.mp4")
        vf = (
            f"fps={fps},"
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1,format=yuv420p"
        )
        cmd = [
            FFMPEG,
            "-y",
            "-threads",
            "1",
            "-i",
            os.path.abspath(src),
            "-vf",
            vf,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-r",
            str(fps),
            "-video_track_timescale",
            "24000",
            "-movflags",
            "+faststart",
            dst,
        ]
        print(f"[ffmpeg] Normalizing clip {idx + 1}/{len(clip_paths)} -> {width}x{height}: {os.path.basename(src)}")
        subprocess.run(cmd, check=True)
        normalized_paths.append(dst)

    list_file = os.path.join(temp_dir, "_concat_list.txt")
    with open(list_file, "w") as f:
        for path in normalized_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")

    cmd = [
        FFMPEG,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_file,
        "-c",
        "copy",
        "-r",
        str(fps),
        output_path,
    ]
    print(f"[ffmpeg] Concatenating {len(normalized_paths)} normalized clips")
    subprocess.run(cmd, check=True)
    shutil.rmtree(temp_dir)
    return output_path


def main():
    workspace = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WORKSPACE
    with open(os.path.join(workspace, "gen_input.json"), "r") as f:
        gen_input = json.load(f)

    clip_entries = []
    missing_clips = []
    print(f"[Component 6] Expecting {len(gen_input['downbeat_frame']) - 1} segment pairs based on gen_input.json.")
    for i in range(len(gen_input["downbeat_frame"]) - 1):
        i2v_path = os.path.join(workspace, f"seg{i:03d}_i2v.mp4")
        trans_path = os.path.join(workspace, f"seg{i:03d}_flf2v.mp4")

        if os.path.exists(i2v_path) and os.path.getsize(i2v_path) > 1024:
            clip_entries.append({"path": i2v_path, "segment": i, "kind": "i2v"})
            print(f"[Component 6] Adding i2v clip: {i2v_path}")
        else:
            print(f"[Component 6] MISSING or invalid i2v clip, skipping: {i2v_path}")
            missing_clips.append(i2v_path)

        if os.path.exists(trans_path) and os.path.getsize(trans_path) > 1024:
            clip_entries.append({"path": trans_path, "segment": i, "kind": "flf2v"})
            print(f"[Component 6] Adding transition clip: {trans_path}")
        else:
            print(f"[Component 6] No valid transition clip found for segment {i}: {trans_path} (will skip)")

    if missing_clips:
        print("[Component 6] WARNING: Missing required clip files. Continuing with available clips:")
        for p in missing_clips:
            print(f"  - {p}")

    if not clip_entries:
        print("[Component 6] ERROR: No valid clips found. Nothing to concatenate.")
        sys.exit(1)

    clip_paths = [entry["path"] for entry in clip_entries]
    first_entry = clip_entries[0]
    first_segment = first_entry["segment"]
    if first_entry["kind"] == "flf2v":
        first_start_sec = (
            float(gen_input["downbeat_frame"][first_segment]["start_sec"])
            + float(gen_input["dynamic_frames"][first_segment]) / float(gen_input["fps"])
        )
    else:
        first_start_sec = float(gen_input["downbeat_frame"][first_segment]["start_sec"])

    is_partial = bool(missing_clips) or first_segment != 0
    print(f"[Component 6] Final clip list contains {len(clip_paths)} files.")
    if is_partial:
        print("[Component 6] Output is partial because one or more clips before/in the sequence are missing.")
        print(f"[Component 6] Partial audio should start at {first_start_sec:.3f}s (segment {first_segment}, {first_entry['kind']}).")
    print("[Component 6] Stitching together clips using FFMPEG...")
    res_path = _ffmpeg_concat_reencode(clip_paths, gen_input["fps"])
    output_name = "final_silent_video_partial.mp4" if is_partial else "final_silent_video.mp4"
    target_path = os.path.join(workspace, output_name)
    shutil.move(res_path, target_path)

    concat_info = {
        "output_name": output_name,
        "is_partial": is_partial,
        "clip_count": len(clip_paths),
        "first_segment": first_segment,
        "first_clip_kind": first_entry["kind"],
        "start_sec": first_start_sec,
        "missing_clips": missing_clips,
        "clips": clip_entries,
    }
    with open(os.path.join(workspace, "concat_info.json"), "w") as f:
        json.dump(concat_info, f, indent=4)
    print(f"[Component 6] Wrote concat metadata -> {os.path.join(workspace, 'concat_info.json')}")
    print(f"[Component 6] Done -> {target_path}")

if __name__ == "__main__":
    main()