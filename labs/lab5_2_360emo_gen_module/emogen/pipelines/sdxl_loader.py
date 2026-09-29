import sys
from diffusers import StableDiffusionXLPipeline


def load_panorama_pipe(
    sdxl_path: str,
    device: str = "cuda",
    torch_dtype=None,
    lora_repo: str = "artificialguybr/360Redmond",
):
    """
    Load SDXL base model and apply panorama LoRA. Returns (pipe, prompt_prefix).
    """
    if torch_dtype is None:
        # keep behavior aligned with the original scripts
        import torch

        torch_dtype = torch.float16 if str(device) != "cpu" else torch.float32

    print(f"Loading SDXL from {sdxl_path}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        sdxl_path,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16",
    ).to(device)

    print(f"Loading panorama LoRA from {lora_repo}...")
    try:
        pipe.load_lora_weights(lora_repo, adapter_name="panorama")
    except IndexError:
        # The UNet weights load successfully before it crashes on the text encoder.
        print(f"Notice: Ignored IndexError when loading text encoder weights for {lora_repo}. (Expected for UNet-only LoRAs)", file=sys.stderr)

    # Trigger phrase used in the original script
    prompt_prefix = "360 view, "
    return pipe, prompt_prefix
