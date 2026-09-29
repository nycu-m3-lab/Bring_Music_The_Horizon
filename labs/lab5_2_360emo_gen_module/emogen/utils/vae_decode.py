"""VAE decode helpers, including circular-padding decode for seamless panoramas."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def decode_with_circular_padding(
    pipe,
    latents: torch.Tensor,
    *,
    panoramic: bool = True,
    latent_pad: int = 8,
) -> Image.Image:
    """Decode latents with the VAE, optionally using circular horizontal padding.

    For panoramic outputs we F.pad the latent horizontally (mode="circular") so the
    VAE decoder convs see a wrapped neighborhood across the seam, then crop the
    matching pixel margin off the decoded image. The VAE is cast to float32 for
    decode stability and restored to float16 after.

    Args:
        pipe: a diffusers SDXL pipeline (uses pipe.vae).
        latents: [1, C, H, W] latent tensor (already divided by scaling_factor).
        panoramic: if True, apply circular pad/crop trick.
        latent_pad: number of latent pixels to pad on each horizontal edge.
                    Image-space crop is ``latent_pad * 8``.

    Returns:
        PIL.Image.Image in uint8 RGB.
    """
    if panoramic:
        latents = F.pad(latents, (latent_pad, latent_pad, 0, 0), mode="circular")

    pipe.vae.to(dtype=torch.float32)
    latents = latents.to(dtype=torch.float32)

    with torch.no_grad():
        image = pipe.vae.decode(latents).sample

    pipe.vae.to(dtype=torch.float16)

    if panoramic:
        crop = latent_pad * 8
        image = image[:, :, :, crop:-crop]

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    return Image.fromarray((image * 255).round().astype(np.uint8))
