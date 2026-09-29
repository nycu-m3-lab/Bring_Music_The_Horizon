import json
import sys
import os

from import_source import DEFAULT_WORKSPACE, mux

def main():
    workspace = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WORKSPACE
    with open(os.path.join(workspace, "gen_input.json"), "r") as f:
        gen_input = json.load(f)

    final_silent = os.path.join(workspace, "final_silent_video.mp4")
    partial_silent = os.path.join(workspace, "final_silent_video_partial.mp4")

    start_sec = float(gen_input["start_sec"])
    concat_info_path = os.path.join(workspace, "concat_info.json")

    if os.path.exists(final_silent):
        silent_video = final_silent
        output_path = os.path.join(workspace, "final_music_video.mp4")
    elif os.path.exists(partial_silent):
        silent_video = partial_silent
        output_path = os.path.join(workspace, "final_music_video_partial.mp4")
        print(f"[Component 7] Using partial silent video: {silent_video}")
        if os.path.exists(concat_info_path):
            with open(concat_info_path, "r") as f:
                concat_info = json.load(f)
            start_sec = float(concat_info.get("start_sec", start_sec))
            print(f"[Component 7] Using partial audio start_sec from concat_info.json: {start_sec:.3f}")
        else:
            print(f"[Component 7] WARNING: concat_info.json not found; falling back to gen_input start_sec={start_sec:.3f}")
    else:
        raise FileNotFoundError(
            f"No silent video found. Expected {final_silent} or {partial_silent}"
        )

    mux(video_path=silent_video, audio_path=gen_input["audio_path"], start_sec=start_sec, output_path=output_path)
    print(f"\n[Component 7] Done. Final Video is at {output_path}")
if __name__ == "__main__":
    main()