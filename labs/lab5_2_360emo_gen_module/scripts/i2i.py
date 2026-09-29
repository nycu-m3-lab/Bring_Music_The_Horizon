#!/usr/bin/env python3
"""360 panorama TPoS-style I2I CLI.

Equivalent to ``Lab5_SEGA/360_i2i.py`` (panorama mode), but built on the
``emogen`` package: pipeline loader, circular padding controller, embeddings,
sampler, decode and annotation are all imported as reusable modules.
"""
import argparse
import os
import subprocess
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from emogen.conditioning.embeddings import prepare_sdxl_embeddings
from emogen.pipelines.circular_padding import create_panorama_circular_padding_controller
from emogen.pipelines.sdxl_loader import load_panorama_pipe
from emogen.sampling.ddim_tpos import TPoS_SDXL_Sampler
from emogen.utils.image_io import encode_init_image_to_latent, load_init_image
from emogen.utils.prompts import sanitize_prompt_for_filename
from emogen.utils.vae_decode import decode_with_circular_padding
from emogen.viz.annotate import annotate_av


def main():
    parser = argparse.ArgumentParser(description="360/Panorama TPoS Native SDXL I2I")

    parser.add_argument("--input_image", type=str, required=True, help="Input image path")
    parser.add_argument("--prompt", type=str, default="mountain landscape")
    parser.add_argument("--sdxl_path", type=str, default="/mnt/M3_Lab/Chiikawa/sdxl_model")
    parser.add_argument("--strength", type=float, default=0.9,
                        help="Strength of the edit (0.0 to 1.0)")

    parser.add_argument("--panorama", default=True, action="store_true",
                        help="Use artificialguybr/360Redmond LoRA")

    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=2048)

    parser.add_argument("--arousal", "-a", type=float, default=-1.5)
    parser.add_argument("--valence", "-v", type=float, default=-1.5)
    parser.add_argument("--no_emo", action="store_true",
                        help="Skip emotion residual (plain I2I)")

    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--edit_guidance_scale", type=float, default=30.0)
    parser.add_argument("--residual_multiplier", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--momentum_scale", type=float, default=0.3)
    parser.add_argument("--mom_beta", type=float, default=0.6)

    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--residual_seed", type=int, default=0)

    parser.add_argument("--circular_last_k_steps", type=int, default=20,
                        help="Apply circular conv padding only in last K steps (panorama)")

    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--heatmap_dir", type=str, default="heatmaps_360_i2i")
    parser.add_argument("--no_annotation", action="store_true")

    args = parser.parse_args()

    if args.panorama:
        args.height, args.width = 800, 1600
        args.no_annotation = True
        if args.heatmap_dir == "heatmaps_360_i2i":
            args.heatmap_dir = "heatmaps_360_i2i_panorama"

    if args.no_emo:
        args.no_annotation = True

    panoramic = args.panorama

    if args.output is None:
        sanitized = sanitize_prompt_for_filename(args.prompt)
        init_name = os.path.splitext(os.path.basename(args.input_image))[0]
        mode = "_panorama" if args.panorama else ""
        if args.no_emo:
            args.output = (
                f"outputs/i2i_no_emo/"
                f"i2i_{init_name}_{sanitized}_str{args.strength}_seed{args.seed}{mode}.png"
            )
        else:
            args.output = (
                f"outputs/i2i_emo/"
                f"i2i_{init_name}_{sanitized}"
                f"_a{args.arousal}_v{args.valence}"
                f"_str{args.strength}_seed{args.seed}{mode}.png"
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print("=== 360/Panorama TPoS I2I ===")
    print(f"  Init image : {args.input_image}")
    print(f"  Prompt     : {args.prompt}")
    print(f"  Strength   : {args.strength}")
    mode_str = "panorama" if args.panorama else "standard"
    print(f"  Mode       : {mode_str}  ({args.width}x{args.height})")
    if not args.no_emo:
        print(f"  Arousal    : {args.arousal}   Valence: {args.valence}")
    print(f"  Output     : {args.output}")

    # 1. Generate Residual via lab4_emoticrafter subprocess
    lab4_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../lab4_emoticrafter"))
    get_residual_script = os.path.join(lab4_path, "get_residual.py")
    residuals_dir = os.path.join(lab4_path, "residuals")

    residual_path = None
    if not args.no_emo:
        print("\n=== Step 1: Generating Residuals ===")
        cmd = [
            "python", get_residual_script,
            "--prompt", args.prompt,
            "--sdxl_path", args.sdxl_path,
            "--seed", str(args.residual_seed),
            "--mode", "single",
            "--values", str(args.arousal), str(args.valence),
        ]
        subprocess.run(cmd, cwd=lab4_path, check=True)

        residual_filename = (
            f"{args.prompt.replace(' ', '_')}"
            f"_a{float(args.arousal)}_v{float(args.valence)}"
            f"_seed{args.residual_seed}.pt"
        )
        residual_path = os.path.join(residuals_dir, residual_filename)

    # 2. Setup Pipeline
    device = "cuda"
    torch.manual_seed(args.seed)

    print("\n=== Step 2: Loading Pipeline ===")
    pipe, prompt_prefix = load_panorama_pipe(args.sdxl_path, device=device, torch_dtype=torch.float16)
    full_prompt = prompt_prefix + args.prompt

    padding_ctl = create_panorama_circular_padding_controller(pipe) if panoramic else None

    # 3. Load Resources
    residual_embeds = None
    if residual_path is not None:
        if os.path.exists(residual_path):
            print("Loading residual...")
            residual_embeds = torch.load(residual_path, map_location=device).to(dtype=torch.float16)
            if residual_embeds.ndim == 2:
                residual_embeds = residual_embeds.unsqueeze(0)
        else:
            print(f"WARNING: Residual not found at {residual_path}, generating without emotion.")

    print("Loading input image...")
    init_image = load_init_image(args.input_image, target_size=(args.width, args.height))
    init_latents = encode_init_image_to_latent(pipe, init_image, device=device)

    print("Encoding text...")
    prompt_emb, neg_emb, pool_emb, neg_pool_emb = prepare_sdxl_embeddings(
        pipe, full_prompt, "", device
    )

    # 4. Sampler
    sampler = TPoS_SDXL_Sampler(pipe, device=device)

    t_enc = int(args.strength * args.num_steps)
    start_step = args.num_steps - t_enc

    # 5. Sample
    print("\n=== Step 3: TPoS DDIM Sampling ===")
    final_latents = sampler.sample(
        latents=init_latents,
        prompt_embeds=prompt_emb,
        negative_prompt_embeds=neg_emb,
        pooled_embeds=pool_emb,
        negative_pooled_embeds=neg_pool_emb,
        residual_embeds=residual_embeds,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        edit_guidance_scale=args.edit_guidance_scale,
        edit_threshold=args.threshold,
        edit_momentum_scale=args.momentum_scale,
        edit_mom_beta=args.mom_beta,
        start_step=start_step,
        height=args.height,
        width=args.width,
        residual_multiplier=args.residual_multiplier,
        heatmap_dir=args.heatmap_dir,
        circular_padding_controller=padding_ctl,
        circular_last_k_steps=args.circular_last_k_steps if panoramic else 0,
    )

    # 6. Decode
    print("\n=== Step 4: Decoding ===")
    final_latents = final_latents / pipe.vae.config.scaling_factor
    image = decode_with_circular_padding(pipe, final_latents, panoramic=panoramic)

    if not args.no_annotation:
        annotate_av(image, args.arousal, args.valence)

    image.save(args.output)
    print(f"\n=== Done! Saved to: {args.output} ===")


if __name__ == "__main__":
    main()
