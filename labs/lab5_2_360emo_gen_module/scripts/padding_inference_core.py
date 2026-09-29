"""
Shared SEGA + panorama sampling for padding scripts.
Callers must load the pipe once, then invoke generate_one_sega_panorama_from_pipe per residual.
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn.functional as F
from PIL import Image

from emogen.conditioning.embeddings import prepare_sdxl_embeddings_for_sega_with_residual
from emogen.sampling.denoiser import create_sdxl_denoiser
from emogen.sampling.euler import simple_euler_sampling
from emogen.viz.heatmaps import create_heatmap_grid

from sdxl_sega_sampler import SEGASDXLGuider


@torch.no_grad()
def generate_one_sega_panorama_from_pipe(
    pipe,
    prompt_with_prefix: str,
    negative_prompt: str,
    *,
    residual_path: str,
    seed: int,
    cfg_scale: float,
    num_steps: int,
    height: int,
    width: int,
    residual_multiplier: float,
    edit_guidance_scale: float,
    edit_threshold: float,
    edit_warmup_steps: int,
    edit_cooldown_steps: int | None,
    delayed_edit_start_step: int,
    save_heatmaps: bool,
    heatmap_output_dir: str,
    device: str,
    enable_circular_padding: bool,
    circular_last_k_steps: int,
    padding_ctl,
):
    torch.manual_seed(seed)

    print("Encoding prompts...")
    embeddings = prepare_sdxl_embeddings_for_sega_with_residual(
        pipe,
        prompt=prompt_with_prefix,
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
        circular_padding_controller=padding_ctl,
        circular_last_k_steps=circular_last_k_steps if enable_circular_padding else 0,
    )

    latents = latents / pipe.vae.config.scaling_factor

    latent_pad = 8
    latents = F.pad(latents, (latent_pad, latent_pad, 0, 0), mode="circular")

    needs_cast = pipe.vae.dtype == torch.float16
    if needs_cast:
        pipe.vae.to(dtype=torch.float32)
        latents = latents.to(dtype=torch.float32)
    with torch.no_grad():
        image = pipe.vae.decode(latents).sample
    if needs_cast:
        pipe.vae.to(dtype=torch.float16)

    crop_size = latent_pad * 8
    image = image[:, :, :, crop_size:-crop_size]

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
