"""TPoS-style native DDIM sampler for SDXL with optional emotion residual guidance.

Ported from ``Lab5_SEGA/360_i2i.py``. The sampler:
  1. Runs CFG denoising on a DDIM trajectory with ``num_inference_steps`` steps.
  2. Optionally applies TPoS edit guidance: a residual embedding shifts the prompt
     embeds, the resulting noise prediction is differenced against the conditional
     branch, sparsified by per-channel quantile threshold, scaled by
     ``edit_guidance_scale``, and EMA-accumulated into a momentum vector ``v_t``.
     The momentum (scaled by ``edit_momentum_scale``) is added to the final eps.
  3. Optionally toggles a circular padding controller on for the last K steps so
     panorama seams stay seamless without forcing it across all steps.

Heatmaps of the sparsified edit delta and the actual TPoS contribution per step
are written via :func:`emogen.viz.heatmaps.save_tpos_heatmap_grid`.
"""
import os

import torch

from diffusers import DDIMScheduler

from ..viz.heatmaps import save_tpos_heatmap_grid


class TPoS_SDXL_Sampler:
    def __init__(self, pipe, device: str = "cuda"):
        self.pipe = pipe
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler
        self.device = device

        if not isinstance(self.scheduler, DDIMScheduler):
            print("Warning: Replacing scheduler with DDIMScheduler")
            self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)

    def get_time_ids(self, height: int, width: int) -> torch.Tensor:
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=torch.float32).to(self.device)
        return add_time_ids

    @torch.no_grad()
    def stochastic_encode(self, latents: torch.Tensor, t, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(latents)

        alpha_t = self.alphas_cumprod[t]
        sqrt_alpha_t = (alpha_t ** 0.5).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = ((1 - alpha_t) ** 0.5).view(-1, 1, 1, 1)

        return sqrt_alpha_t * latents + sqrt_one_minus_alpha_t * noise

    @torch.no_grad()
    def sample(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
        negative_pooled_embeds: torch.Tensor,
        residual_embeds: torch.Tensor = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        edit_guidance_scale: float = 20.0,
        edit_threshold: float = 0.95,
        edit_momentum_scale: float = 0.3,
        edit_mom_beta: float = 0.6,
        start_step: int = 0,
        height: int = 1024,
        width: int = 1024,
        residual_multiplier: float = 1.0,
        heatmap_dir: str = "heatmaps",
        circular_padding_controller=None,
        circular_last_k_steps: int = 0,
    ) -> torch.Tensor:
        if heatmap_dir:
            os.makedirs(heatmap_dir, exist_ok=True)

        heatmap_store = []

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        if start_step > 0:
            timesteps = timesteps[start_step:]
            print(f"I2I: Starting from step {start_step} (t={timesteps[0].item()})")
            t_start = timesteps[0]
            latents = self.stochastic_encode(latents, t_start.long())

        add_time_ids = self.get_time_ids(height, width)
        v_t = torch.zeros_like(latents)

        total_steps = len(timesteps)
        enable_from = max(0, total_steps - int(circular_last_k_steps or 0))
        if circular_padding_controller is not None:
            circular_padding_controller.disable()

        print(f"Running TPoS Native DDIM Loop ({total_steps} steps)...")

        for i, t in enumerate(timesteps):
            if circular_padding_controller is not None and i == enable_from:
                circular_padding_controller.enable()

            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = latent_model_input.to(dtype=torch.float16)

            current_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            current_pooled_embeds = torch.cat([negative_pooled_embeds, pooled_embeds])
            current_time_ids = torch.cat([add_time_ids] * 2)

            added_cond_kwargs = {"text_embeds": current_pooled_embeds, "time_ids": current_time_ids}

            with torch.autocast("cuda", dtype=torch.float16):
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=current_prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                cfg_guidance = guidance_scale * (noise_pred_text - noise_pred_uncond)

                tpos_guidance = torch.zeros_like(noise_pred_uncond)

                if residual_embeds is not None:
                    if residual_embeds.shape[1] == prompt_embeds.shape[1]:
                        edit_prompt_embeds = prompt_embeds + (residual_embeds * residual_multiplier)
                    else:
                        edit_prompt_embeds = prompt_embeds

                    edit_added_cond = {"text_embeds": pooled_embeds, "time_ids": add_time_ids}

                    noise_pred_edit = self.unet(
                        self.scheduler.scale_model_input(latents, t).to(dtype=torch.float16),
                        t,
                        encoder_hidden_states=edit_prompt_embeds,
                        added_cond_kwargs=edit_added_cond,
                        return_dict=False,
                    )[0]

                    delta = noise_pred_edit - noise_pred_text

                    abs_delta = torch.abs(delta)
                    flat_delta = abs_delta.flatten(start_dim=2)
                    threshold_val = torch.quantile(
                        flat_delta.float(), edit_threshold, dim=2, keepdim=True
                    ).to(dtype=delta.dtype)
                    threshold_val = threshold_val.unsqueeze(-1)

                    mask = (abs_delta >= threshold_val).to(dtype=delta.dtype)
                    delta_sparse = delta * mask

                    delta_scaled = delta_sparse * edit_guidance_scale
                    v_t = edit_mom_beta * v_t + (1 - edit_mom_beta) * delta_scaled
                    tpos_guidance = edit_momentum_scale * v_t

                    if heatmap_dir:
                        delta_map = delta_sparse.abs().mean(dim=1).squeeze().float().cpu()
                        guidance_map = tpos_guidance.abs().mean(dim=1).squeeze().float().cpu()
                        heatmap_store.append((i, t.item(), delta_map, guidance_map))

            final_eps = noise_pred_uncond + cfg_guidance + tpos_guidance
            step_output = self.scheduler.step(final_eps, t, latents)
            latents = step_output.prev_sample

            if i % 10 == 0:
                print(f"Step {i}/{total_steps} | t={t.item()} | Mom Mean: {v_t.abs().mean().item():.4f}")

        if circular_padding_controller is not None:
            circular_padding_controller.disable()

        if heatmap_dir and heatmap_store:
            save_tpos_heatmap_grid(heatmap_store, heatmap_dir, target_size=(800, 1600))

        return latents
