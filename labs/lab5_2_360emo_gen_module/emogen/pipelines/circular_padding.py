import torch.nn as nn
import torch.nn.functional as F


class CircularPaddingController:
    """Toggle horizontal circular padding on/off for a set of Conv2d layers."""

    def __init__(self, convs):
        self.convs = convs
        self._orig_forward = {}
        self._enabled = False

    def enable(self):
        if self._enabled:
            return
        for conv in self.convs:
            if conv not in self._orig_forward:
                self._orig_forward[conv] = conv.forward

            def make_circular_forward(c):
                def forward(x):
                    pad_y, pad_x = c.padding
                    if pad_x > 0:
                        x = F.pad(x, (pad_x, pad_x, 0, 0), mode="circular")
                    if pad_y > 0:
                        x = F.pad(x, (0, 0, pad_y, pad_y), mode="constant", value=0)
                    return F.conv2d(
                        x,
                        c.weight,
                        c.bias,
                        c.stride,
                        (0, 0),
                        c.dilation,
                        c.groups,
                    )

                return forward

            conv.forward = make_circular_forward(conv)
        self._enabled = True

    def disable(self):
        if not self._enabled:
            return
        for conv, orig in self._orig_forward.items():
            conv.forward = orig
        self._enabled = False


def _collect_padded_convs(module, out):
    for _, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            if child.padding[0] > 0 or child.padding[1] > 0:
                out.append(child)
        else:
            _collect_padded_convs(child, out)


def create_panorama_circular_padding_controller(pipe, include_vae_decoder: bool = True) -> CircularPaddingController:
    """
    Collect Conv2d layers across the entire UNet (all blocks) and optionally VAE decoder.
    Returns a controller (disabled by default).
    """
    if not hasattr(pipe, "unet"):
        raise AttributeError("Pipeline does not expose `unet`; cannot create circular padding controller.")

    convs = []
    _collect_padded_convs(pipe.unet, convs)

    if include_vae_decoder:
        if not hasattr(pipe, "vae") or not hasattr(pipe.vae, "decoder"):
            raise AttributeError("Pipeline VAE does not expose `decoder`; cannot include VAE decoder in controller.")
        _collect_padded_convs(pipe.vae.decoder, convs)

    return CircularPaddingController(convs)

