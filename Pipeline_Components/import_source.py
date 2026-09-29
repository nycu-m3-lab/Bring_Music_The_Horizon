"""Shared implementation module for the component pipeline."""
import argparse
import glob
import json
import math
import os
import subprocess
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_COMPONENT_DIR, ".."))

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if _COMPONENT_DIR not in sys.path:
    sys.path.insert(0, _COMPONENT_DIR)


# Shared utilities
_DEFAULT_WORKSPACE = os.path.join(_COMPONENT_DIR, "workspace")
DEFAULT_WORKSPACE = _DEFAULT_WORKSPACE


def check_triton_version() -> None:
    """Checks for triton incompatibility with the current torch version."""
    try:
        import triton
        if not hasattr(triton, "language"):
            print("WARNING: Your version of triton is incompatible with the torch version. Disabling triton to prevent crashes.", file=sys.stderr)
            sys.modules["triton"] = None
    except ImportError:
        pass


# Lab0: section extraction
DEFAULT_ALLIN1_PYTHON_PATH = os.environ.get("ALLIN1_PYTHON", "")
DEFAULT_ALLIN1_CONDA_ENV = os.environ.get("ALLIN1_CONDA_ENV", "final_allinone")
DEFAULT_WORKER_SCRIPT = "worker_allinone.py"
TEMP_SECTIONS_JSON = "temp_sections.json"


def _run_allin1_analysis(audio_path: str, allin1_python_path: str, worker_script: str | None = None) -> dict[str, Any]:
    """Run the Allin1 analysis either via a subprocess wrapper or inline import."""
    if worker_script is None:
        worker_script = DEFAULT_WORKER_SCRIPT

    worker_script_path = worker_script
    if not os.path.isabs(worker_script_path):
        worker_script_path = os.path.join(_COMPONENT_DIR, worker_script_path)

    if os.path.exists(worker_script_path) and os.path.isfile(worker_script_path):
        sub_env = os.environ.copy()
        sub_env["ALLIN1_DEVICE"] = os.environ.get("ALLIN1_DEVICE", "cpu")
        temp_sections_path = os.path.join(_COMPONENT_DIR, TEMP_SECTIONS_JSON)
        remove_if_exists(temp_sections_path)
        if allin1_python_path:
            if not os.path.exists(allin1_python_path):
                raise FileNotFoundError(f"Isolated Python not found at {allin1_python_path}")
            cmd = [allin1_python_path, worker_script_path, audio_path]
        else:
            cmd = ["conda", "run", "--no-capture-output", "-n", DEFAULT_ALLIN1_CONDA_ENV, "python", worker_script_path, audio_path]
        subprocess.run(cmd, env=sub_env, cwd=_COMPONENT_DIR, check=True)
        with open(temp_sections_path, "r") as f:
            payload = json.load(f)
        os.remove(temp_sections_path)
        return payload

    try:
        from allin1 import analyze  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Allin1 is not available in the current Python environment") from exc

    warnings = None
    try:
        import warnings as py_warnings
        py_warnings.filterwarnings("ignore")
    except Exception:
        pass

    device = os.environ.get("ALLIN1_DEVICE")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    result = analyze(paths=audio_path, visualize=False, sonify=False, keep_byproducts=False, device=device)
    result = result[0] if isinstance(result, list) else result

    sections = [{"start": s.start, "end": s.end, "label": s.label} for s in result.segments]
    downbeats = list(result.downbeats) if hasattr(result, "downbeats") else []
    bpm = float(result.bpm) if hasattr(result, "bpm") and result.bpm is not None else None
    return {"sections": sections, "downbeats": downbeats, "bpm": bpm}


def extract_sections(audio_path: str, allin1_python_path: str = DEFAULT_ALLIN1_PYTHON_PATH, worker_script: str = DEFAULT_WORKER_SCRIPT):
    """Run allin1 and return (sections, downbeats, bpm)."""
    payload = _run_allin1_analysis(audio_path, allin1_python_path, worker_script=worker_script)
    if isinstance(payload, list):
        return payload, [], None
    return payload.get("sections", []), payload.get("downbeats", []), payload.get("bpm")


# Lab3: VA extraction
class VAExtractor:
    def __init__(self, va_model_path: str, device: torch.device):
        from labs.lab3_music_to_cont_va.mert_model import MERTContinuousModel

        self.device = device
        self.va_model_path = va_model_path
        print(" -> Loading MERT VA Regressor...")
        if not os.path.exists(va_model_path):
            raise FileNotFoundError(f"VA Model not found at {va_model_path}")

        self.model = MERTContinuousModel().to(device)
        self.model.load_state_dict(torch.load(va_model_path, map_location=device))
        self.model.eval()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)

    @torch.no_grad()
    def extract_va_tuples(self, audio_path: str, hz: float) -> torch.Tensor:
        import librosa

        sr = 24000
        audio, _ = librosa.load(audio_path, sr=sr)
        target_steps = max(1, round((len(audio) / sr) * hz))
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)
        preds = self.model(input_values)
        preds_transposed = preds.permute(0, 2, 1)
        preds_averaged = F.adaptive_avg_pool1d(preds_transposed, output_size=target_steps)
        return preds_averaged.permute(0, 2, 1).squeeze(0)


def aggregate_va_by_seconds(va_tuples: torch.Tensor, window_sec: float = 2.0, va_hz: float = 2.0) -> torch.Tensor:
    if va_tuples.ndim != 2 or va_tuples.shape[1] != 2:
        raise ValueError(f"Expected va_tuples shape (T, 2), got {tuple(va_tuples.shape)}")
    if window_sec <= 0 or va_hz <= 0:
        raise ValueError("window_sec and va_hz must be > 0")

    frames_per_window = int(round(window_sec * va_hz))
    if frames_per_window <= 0:
        raise ValueError(f"Invalid frames_per_window={frames_per_window}; check window_sec*va_hz")

    T = va_tuples.shape[0]
    num_windows = int(math.ceil(T / frames_per_window))
    aggregated = []
    for w in range(num_windows):
        s = w * frames_per_window
        e = min((w + 1) * frames_per_window, T)
        aggregated.append(va_tuples[s:e].mean(dim=0, keepdim=True))
    return torch.cat(aggregated, dim=0)


# Lab4: emotion residuals
class EmotiCrafter:
    def __init__(self, sdxl_path: str, eit_ckpt_path: str, device: torch.device):
        from diffusers import StableDiffusionXLPipeline
        from transformers import GPT2Config

        from labs.lab4_emoticrafter.model import EmotionInjectionTransformer

        self.device = device
        print(" -> Loading SDXL Pipeline and EmotionInjectionTransformer (EIT)...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(sdxl_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(device)

        config = GPT2Config.from_pretrained(os.path.abspath(os.path.join(_PROJECT_ROOT, "labs", "lab4_emoticrafter", "config")))
        self.eit = EmotionInjectionTransformer(config, final_out_type="Linear+LN").to(device)

        ckpt = torch.load(eit_ckpt_path, map_location=device)
        if list(ckpt.keys())[0].startswith("module."):
            ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

        self.eit.load_state_dict(ckpt)
        self.eit.eval().to(device)

    @torch.no_grad()
    def generate_residuals(self, va_tuples: torch.Tensor, base_prompt: str, seed: int) -> torch.Tensor:
        torch.Generator(device=self.device).manual_seed(seed)
        (prompt_embeds_ori, _negative_prompt_embeds, _pooled_prompt_embeds_ori, _negative_pooled_prompt_embeds) = self.pipe.encode_prompt(
            prompt=[base_prompt], prompt_2=[base_prompt], device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=True
        )

        residuals_list = []
        for i in range(va_tuples.shape[0]):
            v = va_tuples[i, 0].item()
            a = va_tuples[i, 1].item()
            v_tensor = torch.FloatTensor([[v]]).to(self.device)
            a_tensor = torch.FloatTensor([[a]]).to(self.device)
            out = self.eit(inputs_embeds=prompt_embeds_ori.to(torch.float32), arousal=a_tensor, valence=v_tensor)
            residuals_list.append(out[0] - prompt_embeds_ori)
        return torch.cat(residuals_list, dim=0)


# Lab5: panorama generation
LAB5_2_DIR = os.path.join(_PROJECT_ROOT, "labs", "lab5_2_360emo_gen_module")
LAB5_2_SCRIPTS_DIR = os.path.join(LAB5_2_DIR, "scripts")


def _ensure_lab5_paths() -> None:
    for p in (LAB5_2_DIR, LAB5_2_SCRIPTS_DIR):
        if p not in sys.path:
            sys.path.append(p)


def _dump_tensor_to_temp_pt(tensor: torch.Tensor) -> str:
    fd, path = tempfile.mkstemp(suffix=".pt", prefix="residual_")
    os.close(fd)
    torch.save(tensor, path)
    return path


def _load_residual_stack(path_or_tensor: Any) -> torch.Tensor:
    if isinstance(path_or_tensor, torch.Tensor):
        t = path_or_tensor
    elif isinstance(path_or_tensor, str):
        if path_or_tensor.endswith(".npy"):
            import numpy as np
            t = torch.from_numpy(np.load(path_or_tensor))
        else:
            t = torch.load(path_or_tensor, map_location="cpu")
    else:
        raise TypeError(f"Unsupported residual source: {type(path_or_tensor)}")

    if t.ndim == 2:
        t = t.unsqueeze(0)
    if t.ndim != 3:
        raise ValueError(f"Expected residual stack of shape (T, tokens, dim); got {tuple(t.shape)}")
    return t


class PanoramaT2I:
    def __init__(self, sdxl_path: str, device: str = "cuda"):
        self.sdxl_path = sdxl_path
        self.device = device
        _ensure_lab5_paths()
        from emogen.pipelines.sdxl_loader import load_panorama_pipe
        print(" -> Loading SDXL panorama pipeline (T2I)...")
        self.pipe, self.prompt_prefix = load_panorama_pipe(sdxl_path, device=device)

    @torch.no_grad()
    def generate(self, prompt: str, negative_prompt: str, seed: int = 42, height: int = 800, width: int = 1600, num_steps: int = 50,
                 cfg_scale: float = 7.5, residual_path: str | None = None, residual_tensor: torch.Tensor | None = None,
                 residual_multiplier: float = 1.0, edit_guidance_scale: float = 25.0, edit_threshold: float = 0.75,
                 edit_warmup_steps: int = 0, edit_cooldown_steps: int | None = 30, delayed_edit_start_step: int = 3,
                 circular_last_k_steps: int = 20, annotate: bool = True, arousal: float = 0.0, valence: float = 0.0,
                 save_heatmaps: bool = False, heatmap_output_dir: str = "sega_heatmaps", output_path: str | None = None) -> Image.Image:
        from emogen.viz.annotate import annotate_av

        cleanup_residual_path: str | None = None
        if residual_tensor is not None and residual_path is None:
            residual_path = _dump_tensor_to_temp_pt(residual_tensor)
            cleanup_residual_path = residual_path
        try:
            if residual_path is None:
                image = self._generate_plain(prompt=prompt, negative_prompt=negative_prompt, seed=seed, height=height, width=width,
                                            num_steps=num_steps, cfg_scale=cfg_scale)
                annotate = False
            else:
                image = self._generate_sega(prompt=prompt, negative_prompt=negative_prompt, cfg_scale=cfg_scale, num_steps=num_steps,
                                            seed=seed, height=height, width=width, residual_path=residual_path,
                                            residual_multiplier=residual_multiplier, edit_guidance_scale=edit_guidance_scale,
                                            edit_threshold=edit_threshold, edit_warmup_steps=edit_warmup_steps,
                                            edit_cooldown_steps=edit_cooldown_steps, delayed_edit_start_step=delayed_edit_start_step,
                                            save_heatmaps=save_heatmaps, heatmap_output_dir=heatmap_output_dir,
                                            circular_last_k_steps=circular_last_k_steps)
        finally:
            if cleanup_residual_path is not None and os.path.exists(cleanup_residual_path):
                os.remove(cleanup_residual_path)

        if annotate:
            annotate_av(image, arousal, valence)

        if output_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
            image.save(output_path)
            print(f"[PanoramaT2I] Saved: {output_path}")
        return image

    @torch.no_grad()
    def generate_batch(self, residuals, prompt: str, negative_prompt: str, output_dir: str, va_tuples: torch.Tensor | None = None,
                      seeds: list[int] | int = 42, prefix: str = "seg", annotate: bool = True, **generate_kwargs) -> list[Image.Image]:
        residual_stack = _load_residual_stack(residuals)
        T = residual_stack.shape[0]
        if isinstance(seeds, int):
            seed_list = [seeds] * T
        else:
            if len(seeds) != T:
                raise ValueError(f"seeds length {len(seeds)} != residuals T {T}")
            seed_list = list(seeds)

        os.makedirs(output_dir, exist_ok=True)
        images: list[Image.Image] = []
        for i in range(T):
            slice_i = residual_stack[i]
            v = va_tuples[i, 0].item() if va_tuples is not None else 0.0
            a = va_tuples[i, 1].item() if va_tuples is not None else 0.0
            out_path = os.path.join(output_dir, f"{prefix}_{i:03d}.png")
            print(f"\n[PanoramaT2I][{i+1}/{T}] valence={v:.3f} arousal={a:.3f} -> {out_path}")
            img = self.generate(prompt=prompt, negative_prompt=negative_prompt, seed=seed_list[i], residual_tensor=slice_i, arousal=a,
                                valence=v, annotate=annotate, output_path=out_path, **generate_kwargs)
            images.append(img)
        return images

    @torch.no_grad()
    def _generate_plain(self, prompt: str, negative_prompt: str, seed: int, height: int, width: int, num_steps: int, cfg_scale: float) -> Image.Image:
        from emogen.pipelines.circular_padding import create_panorama_circular_padding_controller
        generator = torch.Generator(device=self.device).manual_seed(seed)
        full_prompt = f"{self.prompt_prefix}{prompt}"
        print(f"Using panorama LoRA with prompt: {full_prompt}")
        padding_ctl = create_panorama_circular_padding_controller(self.pipe)
        padding_ctl.enable()
        try:
            result = self.pipe(prompt=full_prompt, negative_prompt=negative_prompt if negative_prompt else None, height=height, width=width,
                              num_inference_steps=num_steps, guidance_scale=cfg_scale, generator=generator)
        finally:
            if hasattr(padding_ctl, "disable"):
                padding_ctl.disable()
        return result.images[0]

    @torch.no_grad()
    def _generate_sega(self, prompt: str, negative_prompt: str, cfg_scale: float, num_steps: int, seed: int, height: int, width: int,
                      residual_path: str, residual_multiplier: float, edit_guidance_scale: float, edit_threshold: float,
                      edit_warmup_steps: int, edit_cooldown_steps: int | None, delayed_edit_start_step: int, save_heatmaps: bool,
                      heatmap_output_dir: str, circular_last_k_steps: int) -> Image.Image:
        from emogen.conditioning.embeddings import prepare_sdxl_embeddings_for_sega_with_residual
        from emogen.pipelines.circular_padding import create_panorama_circular_padding_controller
        from emogen.sampling.denoiser import create_sdxl_denoiser
        from emogen.sampling.euler import simple_euler_sampling
        from emogen.viz.heatmaps import create_heatmap_grid
        from sdxl_sega_sampler import SEGASDXLGuider

        torch.manual_seed(seed)
        full_prompt = f"{self.prompt_prefix}{prompt}"
        print(f"Using panorama LoRA with prompt: {full_prompt}")
        padding_ctl = create_panorama_circular_padding_controller(self.pipe)
        embeddings = prepare_sdxl_embeddings_for_sega_with_residual(self.pipe, prompt=full_prompt, editing_prompts=[], negative_prompt=negative_prompt,
                                                                    device=self.device, residual_path=residual_path,
                                                                    residual_multiplier=residual_multiplier)
        guider = SEGASDXLGuider(edit_guidance_scale=edit_guidance_scale, edit_threshold=edit_threshold,
                                edit_warmup_steps=edit_warmup_steps, edit_cooldown_steps=edit_cooldown_steps,
                                edit_momentum_scale=0.3, edit_mom_beta=0.6, unconditional_guidance_scale=cfg_scale,
                                delayed_edit_start_step=delayed_edit_start_step, save_heatmaps=save_heatmaps)
        denoiser = create_sdxl_denoiser(self.pipe, height=height, width=width)
        latents_shape = (1, self.pipe.unet.config.in_channels, height // 8, width // 8)
        latents = torch.randn(latents_shape, device=self.device, dtype=torch.float16)
        latents = simple_euler_sampling(denoiser=denoiser, guider=guider, x=latents, cond=embeddings["cond"], uc=embeddings["uc"],
                                        num_steps=num_steps, circular_padding_controller=padding_ctl,
                                        circular_last_k_steps=circular_last_k_steps)
        latents = latents / self.pipe.vae.config.scaling_factor
        latent_pad = 8
        latents = F.pad(latents, (latent_pad, latent_pad, 0, 0), mode="circular")
        needs_cast = self.pipe.vae.dtype == torch.float16
        if needs_cast:
            self.pipe.vae.to(dtype=torch.float32)
            latents = latents.to(dtype=torch.float32)
        image = self.pipe.vae.decode(latents).sample
        if needs_cast:
            self.pipe.vae.to(dtype=torch.float16)
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
                create_heatmap_grid(heatmaps, os.path.join(heatmap_output_dir, "heatmap_grid_edit_range.png"), target_size=grid_target_size, key="edit_range")
                create_heatmap_grid(heatmaps, os.path.join(heatmap_output_dir, "heatmap_grid_psi.png"), target_size=grid_target_size, key="psi")
            else:
                print("No heatmaps collected (editing may not have been active).")
        return image


class PanoramaI2I:
    def __init__(self, sdxl_path: str, device: str = "cuda"):
        self.sdxl_path = sdxl_path
        self.device = device
        _ensure_lab5_paths()
        from emogen.pipelines.sdxl_loader import load_panorama_pipe
        from emogen.sampling.ddim_tpos import TPoS_SDXL_Sampler
        print(" -> Loading SDXL panorama pipeline + TPoS sampler...")
        self.pipe, self.prompt_prefix = load_panorama_pipe(sdxl_path, device=device, torch_dtype=torch.float16)
        self.sampler = TPoS_SDXL_Sampler(self.pipe, device=device)

    @torch.no_grad()
    def generate(self, input_image, prompt: str, negative_prompt: str = "", residual_path: str | None = None,
                 residual_tensor: torch.Tensor | None = None, strength: float = 0.9, seed: int = 42, height: int = 800,
                 width: int = 1600, num_steps: int = 50, guidance_scale: float = 5.0, edit_guidance_scale: float = 30.0,
                 residual_multiplier: float = 1.0, threshold: float = 0.75, momentum_scale: float = 0.3, mom_beta: float = 0.6,
                 circular_last_k_steps: int = 20, panoramic: bool = True, heatmap_dir: str = "heatmaps_360_i2i_panorama",
                 annotate: bool = False, arousal: float = 0.0, valence: float = 0.0, output_path: str | None = None) -> Image.Image:
        from emogen.conditioning.embeddings import prepare_sdxl_embeddings
        from emogen.pipelines.circular_padding import create_panorama_circular_padding_controller
        from emogen.utils.image_io import encode_init_image_to_latent, load_init_image
        from emogen.utils.vae_decode import decode_with_circular_padding
        from emogen.viz.annotate import annotate_av

        torch.manual_seed(seed)
        full_prompt = self.prompt_prefix + prompt
        padding_ctl = create_panorama_circular_padding_controller(self.pipe) if panoramic else None
        residual_embeds = None
        if residual_tensor is not None:
            residual_embeds = residual_tensor.to(device=self.device, dtype=torch.float16)
            if residual_embeds.ndim == 2:
                residual_embeds = residual_embeds.unsqueeze(0)
        elif residual_path is not None:
            if os.path.exists(residual_path):
                print(f"Loading residual from {residual_path}...")
                residual_embeds = torch.load(residual_path, map_location=self.device).to(dtype=torch.float16)
                if residual_embeds.ndim == 2:
                    residual_embeds = residual_embeds.unsqueeze(0)
            else:
                print(f"WARNING: Residual not found at {residual_path}; running without emotion.")

        if isinstance(input_image, str):
            init_image = load_init_image(input_image, target_size=(width, height))
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            try:
                input_image.save(tmp.name)
                init_image = load_init_image(tmp.name, target_size=(width, height))
            finally:
                os.remove(tmp.name)

        init_latents = encode_init_image_to_latent(self.pipe, init_image, device=self.device)
        prompt_emb, neg_emb, pool_emb, neg_pool_emb = prepare_sdxl_embeddings(self.pipe, full_prompt, "", self.device)
        t_enc = int(strength * num_steps)
        start_step = num_steps - t_enc
        final_latents = self.sampler.sample(latents=init_latents, prompt_embeds=prompt_emb, negative_prompt_embeds=neg_emb,
                                            pooled_embeds=pool_emb, negative_pooled_embeds=neg_pool_emb,
                                            residual_embeds=residual_embeds, num_inference_steps=num_steps,
                                            guidance_scale=guidance_scale, edit_guidance_scale=edit_guidance_scale,
                                            edit_threshold=threshold, edit_momentum_scale=momentum_scale, edit_mom_beta=mom_beta,
                                            start_step=start_step, height=height, width=width,
                                            residual_multiplier=residual_multiplier, heatmap_dir=heatmap_dir,
                                            circular_padding_controller=padding_ctl,
                                            circular_last_k_steps=circular_last_k_steps if panoramic else 0)
        final_latents = final_latents / self.pipe.vae.config.scaling_factor
        image = decode_with_circular_padding(self.pipe, final_latents, panoramic=panoramic)
        if annotate:
            annotate_av(image, arousal, valence)
        if output_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
            image.save(output_path)
            print(f"[PanoramaI2I] Saved: {output_path}")
        return image

    @torch.no_grad()
    def generate_batch(self, residuals, prompt: str, output_dir: str, init_image, va_tuples: torch.Tensor | None = None,
                      seeds: list[int] | int = 42, prefix: str = "seg", chain: bool = False, annotate: bool = False,
                      **generate_kwargs) -> list[Image.Image]:
        residual_stack = _load_residual_stack(residuals)
        T = residual_stack.shape[0]
        if isinstance(seeds, int):
            seed_list = [seeds] * T
        else:
            if len(seeds) != T:
                raise ValueError(f"seeds length {len(seeds)} != residuals T {T}")
            seed_list = list(seeds)

        os.makedirs(output_dir, exist_ok=True)
        images: list[Image.Image] = []
        current_init = init_image
        for i in range(T):
            slice_i = residual_stack[i]
            v = va_tuples[i, 0].item() if va_tuples is not None else 0.0
            a = va_tuples[i, 1].item() if va_tuples is not None else 0.0
            out_path = os.path.join(output_dir, f"{prefix}_{i:03d}.png")
            print(f"\n[PanoramaI2I][{i+1}/{T}] valence={v:.3f} arousal={a:.3f} -> {out_path}")
            img = self.generate(input_image=current_init, prompt=prompt, residual_tensor=slice_i, seed=seed_list[i], arousal=a,
                                valence=v, annotate=annotate, output_path=out_path, **generate_kwargs)
            images.append(img)
            if chain:
                current_init = img
        return images


# Utility helpers
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "/mnt/M3_Lab/Chiikawa/ffmpeg/ffmpeg"
FFPROBE = os.environ.get("FFPROBE") or shutil.which("ffprobe") or "/mnt/M3_Lab/Chiikawa/ffmpeg/ffprobe"


def clean_intermediate_files(workspace: str):
    """Removes heavy temporary artifacts like unneeded buffers and embeddings."""
    patterns = [
        "args.json",
        "gen_input.json",
        "keyframe_*.png",
        "seg*.mp4",
        "temp_snippet.wav",
        "va_tuples.pt",
        "raw_residuals.pt",
        "_last_frame_seg*.png",
        "final_silent_video.mp4",
    ]
    for pattern in patterns:
        for path in glob.glob(os.path.join(workspace, pattern)):
            os.remove(path)
            print(f"Cleaned up {path}")


def show_va_tuples(workspace: str):
    """Shows all the values of va_tuples.pt in a human-readable form."""
    va_path = os.path.join(workspace, "va_tuples.pt")
    if not os.path.exists(va_path):
        print(f"File not found: {va_path}")
        return

    va_tuples = torch.load(va_path, map_location="cpu")
    print(f"VA tuples from: {va_path}")
    print("Index | Valence     | Arousal")
    print("------+-------------+-------------")
    for idx, (v, a) in enumerate(va_tuples.tolist()):
        print(f"{idx:5d} | {v:11.6f} | {a:11.6f}")

    values = va_tuples.numpy()
    valence = values[:, 0]
    arousal = values[:, 1]
    print("\nSummary:")
    print(f"  count       : {len(values)}")
    print(f"  valence min : {float(valence.min()):.6f}")
    print(f"  valence max : {float(valence.max()):.6f}")
    print(f"  valence mean: {float(valence.mean()):.6f}")
    print(f"  arousal min : {float(arousal.min()):.6f}")
    print(f"  arousal max : {float(arousal.max()):.6f}")
    print(f"  arousal mean: {float(arousal.mean()):.6f}")

    print("\nTop 5 strongest positive values:")
    top_valence = sorted(valence, reverse=True)[:5]
    top_arousal = sorted(arousal, reverse=True)[:5]
    print(f"  valence: {', '.join(f'{x:.6f}' for x in top_valence)}")
    print(f"  arousal: {', '.join(f'{x:.6f}' for x in top_arousal)}")

    print("\nBottom 5 strongest negative values:")
    bottom_valence = sorted(valence)[:5]
    bottom_arousal = sorted(arousal)[:5]
    print(f"  valence: {', '.join(f'{x:.6f}' for x in bottom_valence)}")
    print(f"  arousal: {', '.join(f'{x:.6f}' for x in bottom_arousal)}")


def remove_if_exists(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def _probe_duration(path: str) -> float:
    """Return media duration in seconds via ffprobe."""
    out = subprocess.check_output(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        text=True,
    ).strip()
    return float(out)


def mux(
    video_path: str,
    audio_path: str,
    start_sec: float,
    output_path: str | None = None,
    audio_bitrate: str = "192k",
) -> str:
    """Mux video_path with audio_path starting at start_sec."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(audio_path)
    if start_sec < 0:
        raise ValueError(f"start_sec must be >= 0, got {start_sec}")

    video_dur = _probe_duration(video_path)
    audio_dur = _probe_duration(audio_path)
    available_audio = audio_dur - start_sec
    print(f"[mux_audio] video duration  = {video_dur:.3f}s")
    print(f"[mux_audio] audio duration  = {audio_dur:.3f}s  (start={start_sec:.3f}s, available={available_audio:.3f}s)")

    if available_audio <= 0:
        raise ValueError(f"start_sec ({start_sec}s) is beyond audio duration ({audio_dur:.3f}s).")
    if available_audio < video_dur:
        print(
            f"[mux_audio] WARNING: audio after offset ({available_audio:.3f}s) is shorter than video ({video_dur:.3f}s); output will be cut to {available_audio:.3f}s."
        )

    if output_path is None:
        base, _ = os.path.splitext(video_path)
        output_path = f"{base}_with_audio.mp4"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    cmd = [
        FFMPEG,
        "-y",
        "-i",
        video_path,
        "-ss",
        f"{start_sec}",
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-shortest",
        output_path,
    ]
    print(f"[mux_audio] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[mux_audio] Saved -> {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline utilities.")
    parser.add_argument("workspace", nargs="?", default=DEFAULT_WORKSPACE, help="Path to workspace folder (default: Final_App_As_Components/workspace)")
    parser.add_argument("--show-va", action="store_true", help="Show va_tuples.pt values instead of cleaning files")
    args = parser.parse_args()

    if args.show_va:
        show_va_tuples(args.workspace)
    else:
        clean_intermediate_files(args.workspace)

__all__ = [
    "DEFAULT_WORKSPACE",
    "check_triton_version",
    "DEFAULT_ALLIN1_PYTHON_PATH",
    "DEFAULT_WORKER_SCRIPT",
    "TEMP_SECTIONS_JSON",
    "extract_sections",
    "VAExtractor",
    "aggregate_va_by_seconds",
    "EmotiCrafter",
    "PanoramaT2I",
    "PanoramaI2I",
    "clean_intermediate_files",
    "show_va_tuples",
    "remove_if_exists",
    "mux",
]
