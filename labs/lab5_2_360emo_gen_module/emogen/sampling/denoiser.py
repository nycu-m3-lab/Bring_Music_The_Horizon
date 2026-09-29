import numpy as np
import torch


def create_sdxl_denoiser(pipe, height: int = 1024, width: int = 1024):
    """Wrap SDXL UNet as a denoiser compatible with K-diffusion-style samplers."""

    class SDXLDenoiser:
        def __init__(self, unet, scheduler, vae, height, width):
            self.unet = unet
            self.scheduler = scheduler
            self.vae = vae
            self.time_ids = torch.tensor([[height, width, 0, 0, height, width]], dtype=torch.float16)
            self.scheduler.set_timesteps(1000)
            self.sigmas = self.scheduler.sigmas
            self.timesteps = self.scheduler.timesteps

        def sigma_to_t(self, sigma):
            if isinstance(sigma, torch.Tensor):
                sigma = sigma[0].item() if sigma.numel() > 1 else sigma.item()

            sigmas_np = self.sigmas.cpu().numpy()
            timesteps_np = self.timesteps.cpu().numpy()
            sigmas_np = sigmas_np[: len(timesteps_np)]
            sigmas_np = sigmas_np[::-1]
            timesteps_np = timesteps_np[::-1]

            t = np.interp(sigma, sigmas_np, timesteps_np)
            return torch.tensor(int(round(t)), device=self.time_ids.device, dtype=torch.long)

        def __call__(self, x, sigma, cond):
            timestep = self.sigma_to_t(sigma)
            encoder_hidden_states = cond["c_crossattn"][0]
            batch_size = x.shape[0]
            time_ids = self.time_ids.repeat(batch_size, 1).to(x.device)

            added_cond_kwargs = {"text_embeds": cond["c_adm"], "time_ids": time_ids}

            c_in = 1 / (sigma**2 + 1).sqrt()
            x_input = x * c_in.view(-1, 1, 1, 1)
            x_input = x_input.to(self.unet.dtype)

            noise_pred = self.unet(
                x_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            noise_pred = noise_pred.to(x.dtype)
            denoised = x - sigma.view(-1, 1, 1, 1) * noise_pred
            return denoised

    return SDXLDenoiser(pipe.unet, pipe.scheduler, pipe.vae, height, width)

