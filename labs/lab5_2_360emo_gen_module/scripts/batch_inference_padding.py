#!/usr/bin/env python3
"""
Batch panorama SEGA inference: load SDXL panorama pipe once, loop over residual .pt files.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Any, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch

from emogen.pipelines.circular_padding import create_panorama_circular_padding_controller
from emogen.pipelines.sdxl_loader import load_panorama_pipe
from emogen.utils.prompts import sanitize_prompt_for_filename
from emogen.viz.annotate import annotate_av

from padding_inference_core import generate_one_sega_panorama_from_pipe


def _load_jobs_from_va_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("va_json must be a JSON list of {residual_path} (valence/arousal optional)")
    jobs = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"va_json[{i}] must be an object")
        rp = item.get("residual_path")
        if not rp:
            raise ValueError(f"va_json[{i}] missing residual_path")
        jobs.append(
            {
                "residual_path": os.path.abspath(str(rp)),
                "valence": float(item["valence"]) if "valence" in item else None,
                "arousal": float(item["arousal"]) if "arousal" in item else None,
            }
        )
    return jobs


def _expand_jobs_to_residual_slices(jobs: List[Dict[str, Any]], temp_dir: str) -> List[Dict[str, Any]]:
    expanded_jobs: List[Dict[str, Any]] = []

    for job_idx, job in enumerate(jobs):
        rp = job["residual_path"]
        residuals = torch.load(rp, map_location="cpu")
        if not torch.is_tensor(residuals):
            raise ValueError(f"Residual file must contain a tensor: {rp}")

        if residuals.ndim == 2:
            residuals = residuals.unsqueeze(0)
        elif residuals.ndim < 2:
            raise ValueError(f"Residual tensor must be 2D or 3D+, got shape={tuple(residuals.shape)} from {rp}")

        num_samples = residuals.shape[0]
        print(f"[batch] Loaded residual tensor from {rp}, shape={tuple(residuals.shape)} -> {num_samples} samples")

        for sample_idx in range(num_samples):
            slice_tensor = residuals[sample_idx].detach().cpu()
            slice_path = os.path.join(temp_dir, f"job{job_idx}_sample{sample_idx}.pt")
            torch.save(slice_tensor, slice_path)
            expanded_jobs.append(
                {
                    "residual_path": slice_path,
                    "valence": job["valence"],
                    "arousal": job["arousal"],
                    "sample_idx": sample_idx,
                    "source_residual_path": rp,
                }
            )

    return expanded_jobs


def main():
    parser = argparse.ArgumentParser(description="Batch panorama emotion-guided images (single SDXL load)")
    parser.add_argument("--prompt", type=str, default="beautiful forest")
    parser.add_argument("--sdxl_path", type=str, default="/mnt/M3_Lab/Chiikawa/sdxl_model")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")

    parser.add_argument("--residual_paths", type=str, nargs="+", default=None)
    parser.add_argument(
        "--va_json",
        type=str,
        default=None,
        help="JSON list of {residual_path} (valence/arousal optional); overrides --residual_paths",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--no_annotation", action="store_true")

    parser.add_argument("--edit_guidance_scale", type=float, default=30.0)
    parser.add_argument("--residual_multiplier", type=float, default=1.0)
    parser.add_argument("--delayed_edit_start_step", type=int, default=0)
    parser.add_argument("--edit_cooldown_steps", type=int, default=15)
    parser.add_argument("--edit_warmup_steps", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--edit_threshold", type=float, default=0.75)

    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=1600)

    parser.add_argument("--panorama", action="store_true")
    parser.add_argument("--save_heatmaps", action="store_true")
    parser.add_argument("--heatmap_output_dir", type=str, default=None)
    parser.add_argument("--circular_last_k_steps", type=int, default=20)
    args = parser.parse_args()

    if not args.panorama:
        raise ValueError("This module only supports --panorama.")

    if args.va_json:
        jobs = _load_jobs_from_va_json(args.va_json)
    else:
        if not args.residual_paths:
            raise ValueError("Provide --residual_paths or --va_json")
        if len(args.residual_paths) != 1:
            raise ValueError("For non-JSON mode, provide exactly one residual file in --residual_paths")
        jobs = [
            {
                "residual_path": os.path.abspath(args.residual_paths[0]),
                "valence": None,
                "arousal": None,
            }
        ]

    for j in jobs:
        if not os.path.isfile(j["residual_path"]):
            raise FileNotFoundError(f"Residual not found: {j['residual_path']}")

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    temp_residual_dir = tempfile.mkdtemp(prefix="residual_slices_", dir=args.output_dir)
    expanded_jobs = _expand_jobs_to_residual_slices(jobs, temp_residual_dir)

    print(f"[batch] Loading panorama pipe once ({len(expanded_jobs)} samples)...")
    pipe, prefix = load_panorama_pipe(sdxl_path=args.sdxl_path, device=device)
    prompt = f"{prefix}{args.prompt}"
    print(f"Using panorama LoRA with prompt: {prompt}")

    padding_ctl = create_panorama_circular_padding_controller(pipe)

    sanitized = sanitize_prompt_for_filename(args.prompt)

    try:
        for i, job in enumerate(expanded_jobs):
            rp = job["residual_path"]
            v = job["valence"]
            a = job["arousal"]
            sample_idx = job["sample_idx"]
            img_seed = args.seed
            out_name = f"padding_{sanitized}_idx{i}_sample{sample_idx}_seed{img_seed}_panorama.png"
            out_path = os.path.join(args.output_dir, out_name)

            hm_dir = args.heatmap_output_dir
            if args.save_heatmaps:
                hm_dir = hm_dir or os.path.join(args.output_dir, f"heatmaps_idx{i}_sample{sample_idx}")
                os.makedirs(hm_dir, exist_ok=True)

            print(f"[batch] ({i + 1}/{len(expanded_jobs)}) sample={sample_idx} residual={rp} -> {out_path}")
            image = generate_one_sega_panorama_from_pipe(
                pipe,
                prompt,
                "",
                residual_path=rp,
                seed=img_seed,
                cfg_scale=args.cfg_scale,
                num_steps=args.num_steps,
                height=args.height,
                width=args.width,
                residual_multiplier=args.residual_multiplier,
                edit_guidance_scale=args.edit_guidance_scale,
                edit_threshold=args.edit_threshold,
                edit_warmup_steps=args.edit_warmup_steps,
                edit_cooldown_steps=args.edit_cooldown_steps,
                delayed_edit_start_step=args.delayed_edit_start_step,
                save_heatmaps=args.save_heatmaps,
                heatmap_output_dir=hm_dir or args.output_dir,
                device=device,
                enable_circular_padding=True,
                circular_last_k_steps=args.circular_last_k_steps,
                padding_ctl=padding_ctl,
            )

            if not args.no_annotation and a is not None and v is not None:
                annotate_av(image, a, v)
            image.save(out_path)
            print(f"Saved: {out_path}")
    finally:
        shutil.rmtree(temp_residual_dir, ignore_errors=True)

    print("[batch] Done.")


if __name__ == "__main__":
    main()
