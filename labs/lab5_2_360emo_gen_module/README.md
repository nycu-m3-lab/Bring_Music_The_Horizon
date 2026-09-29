## labs/lab5_2_360emo_gen_module

This folder contains a modularized implementation of the panorama emotion-guided generator
originally prototyped in `Project/Lab5_SEGA/generate_single_emotion_padding.py`.

### Design
- **`emogen/`**: reusable modules (loading, conditioning, sampling, circular padding, viz, utils)
- **`scripts/`**: thin CLI entrypoints

### Entry points
- `scripts/generate_single_emotion_padding.py`: panorama + SEGA with **circular padding enabled**
- `scripts/generate_single_emotion_360.py`: panorama + SEGA with **circular padding disabled** (baseline)

### Notes
- This module set intentionally supports **`--panorama` only** (no `--360`, `--hdri`, `--hack`).
- Residual generation still relies on `../lab4_emoticrafter/get_residual.py` as in the original script.

