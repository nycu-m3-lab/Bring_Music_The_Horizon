# worker_allinone.py
import os
import sys
import json
import warnings

import torch
from allin1 import analyze

warnings.filterwarnings("ignore")

def main():
    if len(sys.argv) < 2:
        print("Error: No song path provided.")
        sys.exit(1)

    song_path = sys.argv[1]

    device = os.environ.get("ALLIN1_DEVICE")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    result = analyze(
        paths=song_path,
        visualize=False,
        sonify=False,
        keep_byproducts=False,
        device=device,
    )
    result = result[0] if isinstance(result, list) else result

    # Export to JSON
    sections = [{"start": s.start, "end": s.end, "label": s.label} for s in result.segments]
    downbeats = list(result.downbeats) if hasattr(result, "downbeats") else []
    bpm = float(result.bpm) if hasattr(result, "bpm") and result.bpm is not None else None
    payload = {"sections": sections, "downbeats": downbeats, "bpm": bpm}
    with open("temp_sections.json", "w") as f:
        json.dump(payload, f)

if __name__ == "__main__":
    main()