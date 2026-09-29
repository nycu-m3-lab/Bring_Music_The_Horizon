# Bring Music The Horizon: Music-Driven 360&deg; Video Generation

## Introduction

This repository turns an input song and a text prompt into a 360 panorama music video. The pipeline extracts musical structure, predicts valence/arousal over time, injects emotion into SDXL prompt embeddings, renders 360 keyframes, generates motion and transition clips with Wan video generation models through Fal.ai, concatenates the clips, and adds the original audio back.

The pipeline is componentized because different stages need different Conda environments:

| Component | Purpose |
| --- | --- |
| 0 | Save run configuration |
| 1 | Extract structure/downbeats with all-in-one wrapper |
| 2 | Extract valence/arousal tuples |
| 3 | Compute EmotiCrafter residual embeddings |
| 4 | Generate 360 keyframe images |
| 5 | Generate dynamic and transition clips |
| 5.5 | Normalize clips to 16:9 |
| 6 | Concatenate clips |
| 7 | Add music/audio |
| all-in-one worker | Run isolated all-in-one analysis for Component 1 |

The pipeline runner is rerunnable. If a run is interrupted, run the same command again and completed stages will be skipped.

## Prerequisites and Setup Guide

Use Ubuntu or a compatible Linux environment with NVIDIA CUDA support. The main GPU environments use PyTorch CUDA 12.8 wheels; the isolated `final_allinone` env stays on PyTorch CUDA 12.1 because its local NATTEN wheel is built for that stack.

Install or have available:

- Conda or Mamba-compatible Conda
- `bash`
- NVIDIA driver compatible with PyTorch CUDA 12.8
- `sudo` access, unless build tools and ffmpeg are already installed

From the repository root:

### 1. Download `big_files`

Download `BMTH_big_files.zip` from Google Drive:

<https://drive.google.com/drive/folders/12k9KWoyrNN27cpo125oznLFlqYkWUgTL>

Put it into the existing `big_files/` folder, and unzip it there:

```bash
unzip big_files/BMTH_big_files.zip -d big_files
```

### 2. Set API Keys

Fal.ai is required for Component 5 video generation:

```bash
export FAL_KEY="your_fal_key"
```

Gemini is optional. Without `GEMINI_API_KEY`, Component 5 uses fallback prompts and does not import `google.genai`.

```bash
export GEMINI_API_KEY="your_gemini_key"
```

### 3. Create Conda Environments

```bash
bash scripts/setup_conda_envs.sh --yes
```

If sudo is unavailable or system packages are already installed:

```bash
bash scripts/setup_conda_envs.sh --yes --skip-apt
```

### 4. Prepare Music and Prompt

Create the workspace config with your song path and scene prompt:

```bash
conda run --no-capture-output -n final_01567 python Pipeline_Components/Component_0_args_saver.py \
  "./path/to/the/song.mp3" \
  "the scene prompt" \
  "the negative prompt" \
  --workspace "./results"
```

### 5. Run the Pipeline

```bash
bash scripts/run_pipeline_components.sh "./results"
```

Resume after interruption with the same command. Force regeneration with:

```bash
bash scripts/run_pipeline_components.sh --force "./results"
```

By default, Component 5.5 normalizes clips and Components 6/7 write into:

```text
./results/normalized_16x9/final_music_video.mp4
```

Skip normalization with:

```bash
bash scripts/run_pipeline_components.sh --no-normalize "./results"
```

## Clean Up

Remove the project Conda environments:

```bash
bash scripts/remove_conda_envs.sh
```

Non-interactive removal:

```bash
bash scripts/remove_conda_envs.sh --yes
```

Also clean Conda package caches:

```bash
bash scripts/remove_conda_envs.sh --yes --clean-cache
```

The cleanup script only targets these project envs:

```text
final_01567
final_2
final_3
final_4
final_allinone
```

## Acknowledgements

We thank the excellent research and open-source works that made this project possible, including [Stable Diffusion XL (SDXL)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [EmotiCrafter](https://github.com/idvxlab/EmotiCrafter), [SEGA](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4ff83037e8d97b2171b2d3e96cb8e677-Abstract-Conference.html), [360Redmond](https://huggingface.co/artificialguybr/360Redmond), [Wan/Wan2.2](https://github.com/Wan-Video/Wan2.2), [MERT](https://huggingface.co/m-a-p/MERT-v1-95M), and [All-In-One Music Structure Analysis](https://github.com/mir-aidj/all-in-one).

We also acknowledge the datasets that support the music emotion understanding components, including [DEAM](https://cvml.unige.ch/databases/DEAM/) and [PMEmo](https://github.com/HuiZhangDB/PMEmo).

## Citation

If you use this code for your research, please cite the following paper:

```bibtex
@misc{tsai2026bringmusichorizonmusicdriven,
  title={Bring Music The Horizon: Music-Driven 360$^\circ$ Video Generation},
  author={Kai Hsu Tsai and Yong Wei Fu and Hung I Yang and Yu-Chih Chen},
  year={2026},
  eprint={2607.13471},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2607.13471}
}
```

## Contacts

- Kai Hsu Tsai: chris.cs12@nycu.edu.tw
- Yong Wei Fu: willyfu0905.cs12@nycu.edu.tw
- Hung I Yang: holdtensec.cs12@nycu.edu.tw
- Yu-Chih Chen: berriechen@nycu.edu.tw

Department of Computer Science, National Yang Ming Chiao Tung University
