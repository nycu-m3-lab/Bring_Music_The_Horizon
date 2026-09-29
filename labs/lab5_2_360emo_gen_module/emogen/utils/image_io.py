"""I/O helpers for I2I init image loading and VAE encoding."""
import numpy as np
import torch
from PIL import Image


def load_init_image(path: str, target_size: tuple) -> torch.Tensor:
    """Load an image from disk, resize, and normalize to [-1, 1].

    Args:
        path: image file path.
        target_size: (width, height) tuple for PIL resize.

    Returns:
        Tensor of shape [1, 3, H, W] in float32, values in [-1, 1].
    """
    image = Image.open(path).convert("RGB")
    image = image.resize(target_size, Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image * 2.0 - 1.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image


def encode_init_image_to_latent(pipe, image: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """VAE-encode an init image into an SDXL latent.

    The VAE is temporarily cast to float32 for numerical stability of the encode,
    then restored to float16. Returned latent is float16, scaled by
    ``pipe.vae.config.scaling_factor``.
    """
    with torch.no_grad():
        pipe.vae.to(dtype=torch.float32)
        image = image.to(device=device, dtype=torch.float32)
        latents = pipe.vae.encode(image).latent_dist.sample() * pipe.vae.config.scaling_factor
        latents = latents.to(dtype=torch.float16)
        pipe.vae.to(dtype=torch.float16)
    return latents
