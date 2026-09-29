import json
import os
import sys

from import_source import DEFAULT_WORKSPACE, extract_sections

def main():
    workspace = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WORKSPACE
    args_path = os.path.join(workspace, "args.json")
    with open(args_path, "r") as f:
        args = json.load(f)

    print(f"[Component 1] Running allin1 extraction on {args['audio_path']}")
    full_sections, full_downbeats, bpm = extract_sections(args["audio_path"])

    args["full_sections"] = full_sections
    args["full_downbeats"] = full_downbeats
    args["bpm"] = bpm
    args["start_sec"] = float(full_downbeats[0])
    args["four_bar_sec"] = float(full_downbeats[4]) - float(full_downbeats[0])
    args["hz"] = 1.0 / args["four_bar_sec"]
    args["max_keyframes"] = (len(full_downbeats) + 3) // 4

    with open(args_path, "w") as f:
        json.dump(args, f, indent=4)
    print(f"[Component 1] Structural data appended to args.json.")
if __name__ == "__main__":
    main()