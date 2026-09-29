#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from emogen.conditioning.embeddings import prepare_sdxl_embeddings_for_sega_with_residual
from emogen.pipelines.sdxl_loader import load_panorama_pipe
from emogen.sampling.denoiser import create_sdxl_denoiser
from emogen.sampling.euler import simple_euler_sampling
from emogen.utils.prompts import prompt_filename_stem
from emogen.viz.annotate import annotate_av
from emogen.viz.heatmaps import create_heatmap_grid

from sdxl_sega_sampler import SEGASDXLGuider


@torch.no_grad()
def generate_plain_panorama(
    sdxl_path: str,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    num_steps: int,
    cfg_scale: float,
    device: str,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    pipe, prefix = load_panorama_pipe(sdxl_path=sdxl_path, device=device)
    prompt = f"{prefix}{prompt}"
    print(f"Using panorama LoRA with prompt: {prompt}")

    result = pipe(
        prompt=prompt,
        negative_prompt=None,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=cfg_scale,
        generator=generator,
    )
    return result.images[0]


@torch.no_grad()
def generate_with_sega_panorama(
    sdxl_path: str,
    prompt: str,
    negative_prompt: str,
    cfg_scale: float,
    num_steps: int,
    seed: int,
    height: int,
    width: int,
    residual_path: str,
    residual_multiplier: float,
    edit_guidance_scale: float,
    edit_threshold: float,
    edit_warmup_steps: int,
    edit_cooldown_steps: int | None,
    delayed_edit_start_step: int,
    save_heatmaps: bool,
    heatmap_output_dir: str,
    device: str,
):
    torch.manual_seed(seed)
    pipe, prefix = load_panorama_pipe(sdxl_path=sdxl_path, device=device)
    prompt = f"{prefix}{prompt}"
    print(f"Using panorama LoRA with prompt: {prompt}")

    print("Encoding prompts...")
    embeddings = prepare_sdxl_embeddings_for_sega_with_residual(
        pipe,
        prompt=prompt,
        editing_prompts=[],
        negative_prompt=negative_prompt,
        device=device,
        residual_path=residual_path,
        residual_multiplier=residual_multiplier,
    )

    print("Creating SEGA guider...")
    guider = SEGASDXLGuider(
        edit_guidance_scale=edit_guidance_scale,
        edit_threshold=edit_threshold,
        edit_warmup_steps=edit_warmup_steps,
        edit_cooldown_steps=edit_cooldown_steps,
        edit_momentum_scale=0.3,
        edit_mom_beta=0.6,
        unconditional_guidance_scale=cfg_scale,
        delayed_edit_start_step=delayed_edit_start_step,
        save_heatmaps=save_heatmaps,
    )

    denoiser = create_sdxl_denoiser(pipe, height=height, width=width)
    latents_shape = (1, pipe.unet.config.in_channels, height // 8, width // 8)
    latents = torch.randn(latents_shape, device=device, dtype=torch.float16)

    latents = simple_euler_sampling(
        denoiser=denoiser,
        guider=guider,
        x=latents,
        cond=embeddings["cond"],
        uc=embeddings["uc"],
        num_steps=num_steps,
        circular_padding_controller=None,
        circular_last_k_steps=0,
    )

    latents = latents / pipe.vae.config.scaling_factor

    # Baseline CLI: no circular pad/crop.
    needs_cast = pipe.vae.dtype == torch.float16
    if needs_cast:
        pipe.vae.to(dtype=torch.float32)
        latents = latents.to(dtype=torch.float32)
    with torch.no_grad():
        image = pipe.vae.decode(latents).sample
    if needs_cast:
        pipe.vae.to(dtype=torch.float16)

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    image = Image.fromarray((image * 255).round().astype("uint8"))

    if save_heatmaps:
        heatmaps = guider.get_heatmaps()
        if heatmaps:
            os.makedirs(heatmap_output_dir, exist_ok=True)
            target_size = (1024, 2048)
            grid_w = 256
            grid_h = int(256 * target_size[0] / target_size[1])
            grid_target_size = (grid_h, grid_w)
            grid_path = os.path.join(heatmap_output_dir, "heatmap_grid_edit_range.png")
            create_heatmap_grid(heatmaps, grid_path, target_size=grid_target_size, key="edit_range")
            grid_psi_path = os.path.join(heatmap_output_dir, "heatmap_grid_psi.png")
            create_heatmap_grid(heatmaps, grid_psi_path, target_size=grid_target_size, key="psi")
        else:
            print("No heatmaps collected (editing may not have been active).")

    return image


def main():
    parser = argparse.ArgumentParser(description="Panorama Emotion-Guided Image (baseline CLI, no circular padding)")
    parser.add_argument("--prompt", type=str, default="beautiful forest")
    parser.add_argument("--sdxl_path", type=str, default="/mnt/M3_Lab/Chiikawa/sdxl_model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--residual_seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)

    parser.add_argument("--valence", "-v", type=float, default=0.0)
    parser.add_argument("--arousal", "-a", type=float, default=0.0)

    parser.add_argument("--edit_guidance_scale", type=float, default=20.0)
    parser.add_argument("--residual_multiplier", type=float, default=1.0)
    parser.add_argument("--delayed_edit_start_step", type=int, default=0)
    parser.add_argument("--edit_cooldown_steps", type=int, default=15)
    parser.add_argument("--edit_warmup_steps", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--edit_threshold", type=float, default=0.75)

    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=1600)

    parser.add_argument("--panorama", action="store_true", help="Enable panorama mode (required in this module).")
    parser.add_argument("--no_emo", action="store_true", help="Skip residual/SEGA; generate plain image.")
    parser.add_argument("--no_annotation", action="store_true")

    parser.add_argument("--save_heatmaps", action="store_true")
    parser.add_argument("--heatmap_output_dir", type=str, default="sega_heatmaps")

    args = parser.parse_args()

    if not args.panorama:
        raise ValueError("This module only supports --panorama.")

    if args.no_emo:
        args.no_annotation = True

    if args.output is None:
        sanitized_prompt = prompt_filename_stem(args.prompt)
        mode_suffix = "_panorama"
        if args.no_emo:
            mode_suffix += "_noemo"
            args.output = f"base_{sanitized_prompt}_seed{args.seed}{mode_suffix}.png"
        else:
            args.output = f"base_{sanitized_prompt}_a{args.arousal}_v{args.valence}_seed{args.seed}{mode_suffix}.png"

    lab4_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lab4_emoticrafter"))
    get_residual_script = os.path.join(lab4_path, "get_residual.py")
    residuals_dir = os.path.join(lab4_path, "residuals")

    residual_path = None
    if not args.no_emo:
        residual_prompt = args.prompt
        cmd = [
            "python",
            get_residual_script,
            "--prompt",
            residual_prompt,
            "--sdxl_path",
            args.sdxl_path,
            "--seed",
            str(args.residual_seed),
            "--mode",
            "single",
            "--arousal",
            str(args.arousal),
            "--valence",
            str(args.valence),
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=lab4_path, check=True)

        sanitized_residual_prompt = prompt_filename_stem(residual_prompt)
        residual_filename = f"{sanitized_residual_prompt}_a{args.arousal}_v{args.valence}_seed{args.residual_seed}.pt"
        residual_path = os.path.join(residuals_dir, residual_filename)
        if not os.path.exists(residual_path):
            raise FileNotFoundError(f"Residual file not found: {residual_path}")

    if args.no_emo:
        image = generate_plain_panorama(
            sdxl_path=args.sdxl_path,
            prompt=args.prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
            device="cuda",
        )
    else:
        image = generate_with_sega_panorama(
            sdxl_path=args.sdxl_path,
            prompt=args.prompt,
            negative_prompt="",
            cfg_scale=args.cfg_scale,
            num_steps=args.num_steps,
            seed=args.seed,
            height=args.height,
            width=args.width,
            residual_path=residual_path,
            residual_multiplier=args.residual_multiplier,
            edit_guidance_scale=args.edit_guidance_scale,
            edit_threshold=args.edit_threshold,
            edit_warmup_steps=args.edit_warmup_steps,
            edit_cooldown_steps=args.edit_cooldown_steps,
            delayed_edit_start_step=args.delayed_edit_start_step,
            save_heatmaps=args.save_heatmaps,
            heatmap_output_dir=args.heatmap_output_dir,
            device="cuda",
        )

    if not args.no_annotation:
        annotate_av(image, args.arousal, args.valence)

    image.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

