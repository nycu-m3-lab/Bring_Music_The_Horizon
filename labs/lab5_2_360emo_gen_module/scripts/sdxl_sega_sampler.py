"""
SDXL + SEGA Sampler Implementation
Integrates Semantic Guidance (SEGA) with SDXL using K-Diffusion sampling architecture.

Key Components:
1. SEGASDXLGuider: Handles SEGA guidance logic with SDXL's dual embedding system
2. Compatible with BaseDiffusionSampler from sdxl_sampling.py
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union


def gaussian_blur_2d(x, kernel_size=3, sigma=1.0):
    """
    Apply Gaussian blur to a 4D tensor [B, C, H, W].
    """
    channels = x.shape[1]
    
    # Create Gaussian kernel
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    
    mean = (kernel_size - 1) / 2.
    variance = sigma**2.
    
    gaussian_kernel = (1./(2.*torch.pi*variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) / \
                          (2*variance)
                      )
    
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    
    # Reshape to [C, 1, K, K] for depthwise conv
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1).to(x.device, dtype=x.dtype)
    
    # Apply padding
    padding = kernel_size // 2
    
    return F.conv2d(x, gaussian_kernel, padding=padding, groups=channels)


def append_dims(x, target_dims):
    """Append dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def to_denoised(x, sigma, d):
    """Converts a Karras ODE derivative to a denoised prediction."""
    return x - sigma * d


class SEGASDXLGuider:
    """
    SEGA Guider for SDXL models.
    
    Handles:
    - SDXL's dual embedding system (concatenated + pooled)
    - SEGA's semantic editing logic (thresholding, momentum, warmup)
    - Classifier-free guidance (CFG)
    
    Args:
        edit_guidance_scale: Scale for semantic guidance (s_e in paper)
        edit_threshold: Threshold for sparsification (λ in paper)  
        edit_warmup_steps: Number of steps before applying semantic guidance (δ in paper)
        edit_cooldown_steps: Number of steps to apply semantic guidance
        edit_momentum_scale: Scale for momentum term
        edit_mom_beta: Beta for momentum exponential moving average
        edit_weights: Weights for multiple editing concepts
        unconditional_guidance_scale: Scale for CFG (w in paper)
        reverse_editing_direction: Whether to reverse editing direction for each concept
    """
    
    def __init__(
        self,
        edit_guidance_scale: Union[float, List[float]] = 2000.0,
        edit_threshold: Union[float, List[float]] = 0.95,
        edit_warmup_steps: Union[int, List[int]] = 5,
        edit_cooldown_steps: Optional[Union[int, List[int]]] = None,
        edit_momentum_scale: float = 0.3,
        edit_mom_beta: float = 0.6,
        edit_weights: Optional[List[float]] = None,
        unconditional_guidance_scale: float = 7.5,
        reverse_editing_direction: Union[bool, List[bool]] = False,
        use_blur: bool = True,
        delayed_edit_start_step: int = 0,
        save_heatmaps: bool = False,
        save_guidance_term_path: Optional[str] = None,
        load_guidance_term_path: Optional[str] = None,
    ):
        self.edit_guidance_scale = edit_guidance_scale
        self.edit_threshold = edit_threshold
        self.edit_warmup_steps = edit_warmup_steps
        self.edit_cooldown_steps = edit_cooldown_steps
        self.edit_momentum_scale = edit_momentum_scale
        self.edit_mom_beta = edit_mom_beta
        self.edit_weights = edit_weights
        self.unconditional_guidance_scale = unconditional_guidance_scale
        self.reverse_editing_direction = reverse_editing_direction
        self.use_blur = use_blur
        self.delayed_edit_start_step = delayed_edit_start_step
        self.save_heatmaps = save_heatmaps
        self.save_guidance_term_path = save_guidance_term_path
        self.load_guidance_term_path = load_guidance_term_path
        
        # State variables
        self.edit_momentum = None
        self.current_step = 0
        self.num_edit_concepts = 0
        self.edit_concepts_metadata = []
        self.heatmaps = []  # List of {'step', 'edit_range', 'psi'}
        self.saved_guidance_terms = []  # List of (step_index, tensor) when saving
        
        # Load pre-computed guidance terms if path provided
        self.loaded_guidance_terms = None
        if load_guidance_term_path and os.path.exists(load_guidance_term_path):
            arr = np.load(load_guidance_term_path)
            self.loaded_guidance_terms = arr  # shape (num_steps, B, C, H, W)
            print(f"Loaded pre-computed guidance terms from {load_guidance_term_path}, shape={arr.shape}")
        
    def prepare_inputs(self, x, sigma, cond, uc):
        """
        Prepare inputs for SDXL UNet with SEGA editing concepts.
        
        Args:
            x: Latent tensor [B, C, H, W]
            sigma: Current noise level [B]
            cond: Conditional embeddings dict with:
                - 'concat': [B, seq_len, hidden_dim] (from text_encoder)
                - 'pooled': [B, pooled_dim] (from text_encoder_2)
                - 'edit_concepts': List of M dicts, each with 'concat' and 'pooled'
            uc: Unconditional embeddings dict with 'concat' and 'pooled'
        
        Returns:
            Tuple of (x_in, sigma_in, cond_in) ready for UNet forward pass
        """
        # Extract editing concepts
        edit_concepts = cond.get('edit_concepts', [])
        self.num_edit_concepts = len(edit_concepts)
        self.edit_concepts_metadata = edit_concepts
        
        # When loading pre-computed guidance, only run uc + cond (no edit-concept UNet passes)
        use_loaded_guidance = self.loaded_guidance_terms is not None
        if use_loaded_guidance:
            batch_multiplier = 2
            concat_embeds = torch.cat([uc['concat'], cond['concat']], dim=0)
            pooled_embeds = torch.cat([uc['pooled'], cond['pooled']], dim=0)
        else:
            batch_multiplier = 2 + self.num_edit_concepts
            concat_embeds = torch.cat([
                uc['concat'],
                cond['concat'],
            ] + [edit['concat'] for edit in edit_concepts], dim=0)
            pooled_embeds = torch.cat([
                uc['pooled'],
                cond['pooled'],
            ] + [edit['pooled'] for edit in edit_concepts], dim=0)
        
        x_in = torch.cat([x] * batch_multiplier, dim=0)
        sigma_in = torch.cat([sigma] * batch_multiplier, dim=0)
        
        # Create conditioning dict for SDXL UNet
        cond_in = {
            'c_crossattn': [concat_embeds],  # SDXL expects list
            'c_adm': pooled_embeds,  # Pooled embeddings for adaptive layer norm
        }
        
        return x_in, sigma_in, cond_in
    
    def __call__(self, x, denoised, sigma):
        """
        Apply SEGA guidance to denoised predictions.
        
        Args:
            x: Noisy latents [B, C, H, W] (single batch)
            denoised: Denoised predictions from UNet [B * (2 + M), C, H, W]
            sigma: Current noise level [B * (2 + M)]
        
        Returns:
            Final guided denoised prediction [B, C, H, W]
        """
        # Split predictions: uc, cond, (edit_concepts when not loading)
        batch_size = x.shape[0]
        use_loaded_guidance = self.loaded_guidance_terms is not None
        num_chunks = 2 if use_loaded_guidance else (2 + self.num_edit_concepts)
        denoised_chunks = denoised.chunk(num_chunks, dim=0)
        denoised_uc = denoised_chunks[0]
        denoised_cond = denoised_chunks[1]
        denoised_edits = list(denoised_chunks[2:]) if not use_loaded_guidance else []
        
        # Get sigma for single batch
        sigma_val = sigma[:batch_size].view(-1, 1, 1, 1)
        
        # === Section 1: Convert to epsilon (noise) space ===
        eps_uc = (x - denoised_uc) / sigma_val
        eps_cond = (x - denoised_cond) / sigma_val
        eps_edits = [(x - d) / sigma_val for d in denoised_edits]
        
        # === Section 2: Compute CFG guidance ===
        cfg_guidance = self.unconditional_guidance_scale * (eps_cond - eps_uc)
        
        # === Use pre-computed (raw) guidance term when loading: replace computed term, then apply scale/threshold/blur/momentum ===
        if use_loaded_guidance:
            step_idx = min(self.current_step, self.loaded_guidance_terms.shape[0] - 1)
            raw_loaded = torch.from_numpy(self.loaded_guidance_terms[step_idx].copy()).to(
                device=cfg_guidance.device, dtype=cfg_guidance.dtype
            )
            if raw_loaded.dim() == 3:
                raw_loaded = raw_loaded.unsqueeze(0)
            # Skip SEGA when before delayed_edit_start_step (same as normal path)
            if self.current_step < self.delayed_edit_start_step:
                eps_final = eps_uc + cfg_guidance
                self.current_step += 1
                return x - sigma_val * eps_final
            # Apply scale (use first concept's scale when loading single aggregated term)
            edit_scale = self._get_param(self.edit_guidance_scale, 0)
            edit_threshold_val = self._get_param(self.edit_threshold, 0)
            shift = raw_loaded * edit_scale
            # Thresholding
            shift_flat = torch.abs(shift).flatten(start_dim=2)
            threshold_val = torch.quantile(shift_flat, edit_threshold_val, dim=2, keepdim=False)
            threshold_val = threshold_val[:, :, None, None]
            mask = torch.abs(shift) >= threshold_val
            shift = torch.where(mask, shift, torch.zeros_like(shift))
            if self.use_blur:
                shift = gaussian_blur_2d(shift, kernel_size=3, sigma=1.0)
            # Momentum
            if self.edit_momentum is None:
                self.edit_momentum = torch.zeros_like(cfg_guidance)
            self.edit_momentum = (
                self.edit_mom_beta * self.edit_momentum + (1 - self.edit_mom_beta) * shift
            )
            final_semantic_guidance = shift + self.edit_momentum_scale * self.edit_momentum
            # Collect heatmaps when loading (same as normal path)
            if self.save_heatmaps:
                edit_range = (shift.abs() > 1e-6).float().mean(dim=1).detach().cpu()
                heatmap_psi = shift.abs().mean(dim=1).detach().cpu()
                self.heatmaps.append({
                    'step': self.current_step,
                    'edit_range': edit_range,
                    'psi': heatmap_psi,
                    'psi_with_momentum': final_semantic_guidance.abs().mean(dim=1).detach().cpu(),
                })
            eps_final = eps_uc + cfg_guidance + final_semantic_guidance
            self.current_step += 1
            return x - sigma_val * eps_final
        
        # === Section 3: Compute SEGA semantic guidance ===
        # Check if we should skip SEGA editing (delayed start)
        if self.current_step < self.delayed_edit_start_step or self.num_edit_concepts == 0:
            # No editing yet or no editing concepts, just return CFG
            eps_final = eps_uc + cfg_guidance
            self.current_step += 1
            return x - sigma_val * eps_final
        
        # Initialize momentum on first call
        if self.edit_momentum is None:
            self.edit_momentum = torch.zeros_like(cfg_guidance)
        
        # Compute semantic shifts for each editing concept
        concept_weights = torch.zeros(
            (self.num_edit_concepts, batch_size),
            device=denoised.device,
            dtype=denoised.dtype
        )
        
        semantic_shifts = torch.zeros(
            (self.num_edit_concepts, *cfg_guidance.shape),
            device=denoised.device,
            dtype=denoised.dtype
        )
        # Raw shifts (before scale/threshold/blur) for saving when save_guidance_term_path is set
        raw_semantic_shifts = torch.zeros(
            (self.num_edit_concepts, *cfg_guidance.shape),
            device=denoised.device,
            dtype=denoised.dtype
        )
        
        warmup_concepts = []
        
        for c_idx, eps_edit in enumerate(eps_edits):
            # Get per-concept parameters
            edit_scale_c = self._get_param(self.edit_guidance_scale, c_idx)
            edit_threshold_c = self._get_param(self.edit_threshold, c_idx)
            reverse_direction_c = self._get_param(self.reverse_editing_direction, c_idx)
            edit_weight_c = self._get_param(self.edit_weights, c_idx, default=1.0)
            warmup_steps_c = self._get_param(self.edit_warmup_steps, c_idx)
            cooldown_steps_c = self._get_param(
                self.edit_cooldown_steps, c_idx, 
                default=self.current_step + 1
            )
            
            # Check if this concept is in warmup or cooldown
            if self.current_step >= warmup_steps_c:
                warmup_concepts.append(c_idx)
            
            if self.current_step >= cooldown_steps_c:
                continue
            
            # === Section 2: Compute semantic shift (ψ_i in paper) ===
            # ψ_i = ε_edit_i - ε_uncond
            # Check if we should use the conditional prompt as the baseline (for residual guidance)
            use_cond_as_neg = False
            concept_multiplier = 1.0
            if self.edit_concepts_metadata and c_idx < len(self.edit_concepts_metadata):
                use_cond_as_neg = self.edit_concepts_metadata[c_idx].get('use_cond_as_negative', False)
                concept_multiplier = self.edit_concepts_metadata[c_idx].get('concept_multiplier', 1.0)
            
            baseline = eps_cond if use_cond_as_neg else eps_uc
            shift = eps_edit - baseline
            
            # Apply concept multiplier (e.g. residual multiplier)
            if concept_multiplier != 1.0:
                shift = shift * concept_multiplier
            
            # Store raw shift (before scale/threshold/blur) for --save_guidance_term
            raw_semantic_shifts[c_idx] = shift
            
            # Debug: Print shift stats before scaling
            if self.current_step % 5 == 0 and c_idx == 0:
                print(f"Step {self.current_step} | Concept {c_idx}: Shift Mean={shift.mean().item():.4f}, Std={shift.std().item():.4f}, Max={shift.max().item():.4f}")
            
            # === Section 3: Apply scaling and thresholding ===
            # Scale: s_e * ψ_i
            shift = shift * edit_scale_c
            
            # Thresholding: Keep only top (1 - λ) values
            # Compute threshold per batch and channel
            shift_flat = torch.abs(shift).flatten(start_dim=2)  # [B, C, H*W]
            threshold_val = torch.quantile(
                shift_flat, 
                edit_threshold_c, 
                dim=2, 
                keepdim=False
            )  # [B, C]
            
            # Apply threshold mask
            threshold_val = threshold_val[:, :, None, None]  # [B, C, 1, 1]
            mask = torch.abs(shift) >= threshold_val
            shift = torch.where(
                mask,
                shift,
                torch.zeros_like(shift)
            )
            
            # Apply Gaussian smoothing to reduce high-frequency artifacts from thresholding
            if self.use_blur:
                shift = gaussian_blur_2d(shift, kernel_size=3, sigma=1.0)
            
            # Debug: Print shift stats after thresholding
            if self.current_step % 5 == 0 and c_idx == 0:
                active_ratio = mask.float().mean().item()
                print(f"Step {self.current_step} | Concept {c_idx}: Post-Threshold Active={active_ratio:.2%}, Mean={shift.mean().item():.4f}, Max={shift.max().item():.4f}")

            # Store shift and weight
            semantic_shifts[c_idx] = shift
            concept_weights[c_idx] = torch.full((batch_size,), edit_weight_c)
        
        # === Section 4: Apply warmup logic ===
        # Only use concepts that have passed warmup
        if len(warmup_concepts) > 0:
            warmup_indices = torch.tensor(warmup_concepts, device=denoised.device)
            
            # Normalize weights for warmup concepts
            warmup_weights = concept_weights[warmup_indices]
            warmup_weights = torch.clamp(warmup_weights, min=0.0)
            weight_sum = warmup_weights.sum(dim=0, keepdim=True)
            weight_sum = torch.where(weight_sum > 0, weight_sum, torch.ones_like(weight_sum))
            warmup_weights = warmup_weights / weight_sum
            
            # Aggregate semantic shifts with weights
            warmup_shifts = semantic_shifts[warmup_indices]
            aggregated_shift = torch.einsum(
                'cb,cbijk->bijk',
                warmup_weights,
                warmup_shifts
            )
            # Raw aggregated (before scale/threshold/blur) for saving
            raw_warmup_shifts = raw_semantic_shifts[warmup_indices]
            raw_aggregated_shift = torch.einsum(
                'cb,cbijk->bijk',
                warmup_weights,
                raw_warmup_shifts
            )
            
            # === Section 5: Apply momentum ===
            # v_t = β * v_{t-1} + (1 - β) * aggregated_shift
            # final_shift = aggregated_shift + scale * v_t
            self.edit_momentum = (
                self.edit_mom_beta * self.edit_momentum +
                (1 - self.edit_mom_beta) * aggregated_shift
            )
            
            final_semantic_guidance = (
                aggregated_shift + 
                self.edit_momentum_scale * self.edit_momentum
            )
            
            # Save heatmaps: edit range (where threshold is applied) and |psi|
            if self.save_heatmaps:
                # edit_range: 1 where |shift| >= threshold (guidance applied), 0 elsewhere
                edit_range = (aggregated_shift.abs() > 1e-6).float().mean(dim=1).detach().cpu()
                heatmap_psi = aggregated_shift.abs().mean(dim=1).detach().cpu()
                self.heatmaps.append({
                    'step': self.current_step,
                    'edit_range': edit_range,
                    'psi': heatmap_psi,
                    'psi_with_momentum': final_semantic_guidance.abs().mean(dim=1).detach().cpu(),
                })
        else:
            # No concepts are warmed up yet, but still accumulate momentum
            # Aggregate all shifts for momentum accumulation
            all_weights = torch.clamp(concept_weights, min=0.0)
            weight_sum = all_weights.sum(dim=0, keepdim=True)
            weight_sum = torch.where(weight_sum > 0, weight_sum, torch.ones_like(weight_sum))
            all_weights = all_weights / weight_sum
            
            aggregated_shift = torch.einsum(
                'cb,cbijk->bijk',
                all_weights,
                semantic_shifts
            )
            
            self.edit_momentum = (
                self.edit_mom_beta * self.edit_momentum +
                (1 - self.edit_mom_beta) * aggregated_shift
            )
            
            final_semantic_guidance = torch.zeros_like(cfg_guidance)
        
        # Save raw guidance term for this step (before scale/threshold/blur/momentum)
        if self.save_guidance_term_path and len(warmup_concepts) > 0:
            self.saved_guidance_terms.append((self.current_step, raw_aggregated_shift.detach().cpu()))
        elif self.save_guidance_term_path and len(warmup_concepts) == 0:
            # No warmup concepts: save zeros so step indices match
            self.saved_guidance_terms.append((self.current_step, torch.zeros_like(cfg_guidance).detach().cpu()))
        
        # === Section 6: Final prediction ===
        # eps_final = eps_uc + cfg_guidance + semantic_guidance
        eps_final = eps_uc + cfg_guidance + final_semantic_guidance
        
        # Convert back to denoised (x0)
        # denoised = x - sigma * eps
        denoised_final = x - sigma_val * eps_final
        
        # Increment step counter (only once per call)
        self.current_step += 1
        
        return denoised_final
    
    def _get_param(self, param, idx, default=None):
        """Helper to get per-concept parameter."""
        if isinstance(param, list):
            return param[idx] if idx < len(param) else (default if default is not None else param[0])
        return param if param is not None else default
    
    def reset(self):
        """Reset state for new sampling run."""
        self.edit_momentum = None
        self.current_step = 0
        self.num_edit_concepts = 0
        self.edit_concepts_metadata = []
        self.heatmaps = []
        self.saved_guidance_terms = []
    
    def get_heatmaps(self):
        """Return collected heatmaps (edit_range and psi per step)."""
        return self.heatmaps
    
    def get_saved_guidance_terms(self):
        """Return list of (step_index, tensor) for saving to npy. Script builds (num_steps, B, C, H, W) and saves."""
        return self.saved_guidance_terms


# ============================================================================
# Example Usage with SDXL Pipeline
# ============================================================================

def _build_edit_concept_text(prompt: str, editing_prompt: str) -> str:
    """
    Build edit concept as full phrase related to original prompt.
    e.g. prompt="a boy", editing_prompt="sad" -> "a sad boy"
    Inserts editing_prompt before the last word of prompt.
    """
    words = prompt.strip().split()
    if not words:
        return editing_prompt.strip()
    if len(words) == 1:
        return f"{editing_prompt.strip()} {words[0]}".strip()
    return " ".join(words[:-1] + [editing_prompt.strip()] + words[-1:])


def prepare_sdxl_embeddings_for_sega(
    pipe,
    prompt: str,
    editing_prompts: List[str],
    negative_prompt: Optional[str] = None,
    device: str = "cuda",
):
    """
    Prepare embeddings for SDXL + SEGA (original: edit = single concept, baseline = uncond).
    
    Args:
        pipe: StableDiffusionXLPipeline instance
        prompt: Main generation prompt
        editing_prompts: List of editing concept prompts (encoded as standalone)
        negative_prompt: Negative prompt for CFG
        device: Device to put embeddings on
    
    Returns:
        Dict with 'cond', 'uc' containing concatenated and pooled embeddings
    """
    # Encode main prompt
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=[prompt],
        prompt_2=[prompt],
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=[negative_prompt] if negative_prompt else None,
        negative_prompt_2=[negative_prompt] if negative_prompt else None,
    )
    
    # Encode editing concepts (standalone: e.g. "sad" -> guidance = ε(sad) - ε(uncond))
    edit_concepts = []
    for edit_prompt in editing_prompts:
        (
            edit_embeds,
            _,
            edit_pooled_embeds,
            _,
        ) = pipe.encode_prompt(
            prompt=[edit_prompt],
            prompt_2=[edit_prompt],
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        edit_concepts.append({
            'concat': edit_embeds,
            'pooled': edit_pooled_embeds,
        })
    
    # Prepare conditioning dict
    cond = {
        'concat': prompt_embeds,
        'pooled': pooled_prompt_embeds,
        'edit_concepts': edit_concepts,
    }
    
    uc = {
        'concat': negative_prompt_embeds,
        'pooled': negative_pooled_prompt_embeds,
    }
    
    return {'cond': cond, 'uc': uc}


def prepare_sdxl_embeddings_for_sega_prompt_based(
    pipe,
    prompt: str,
    editing_prompts: List[str],
    negative_prompt: Optional[str] = None,
    device: str = "cuda",
):
    """
    Prepare embeddings for SDXL + SEGA with prompt-based edit concepts.
    
    Edit concept = full phrase related to original prompt (e.g. prompt="a boy", edit="sad" -> "a sad boy").
    Guidance term = noise(edit_concept) - noise(origin_prompt), i.e. use_cond_as_negative=True.
    
    Args:
        pipe: StableDiffusionXLPipeline instance
        prompt: Main generation prompt (origin_prompt, e.g. "a boy")
        editing_prompts: List of modifiers (e.g. ["sad"]) -> edit_concept = "a sad boy"
        negative_prompt: Negative prompt for CFG
        device: Device to put embeddings on
    
    Returns:
        Dict with 'cond', 'uc', 'edit_concepts' where each edit has use_cond_as_negative=True
    """
    # Encode main prompt (origin_prompt)
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=[prompt],
        prompt_2=[prompt],
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=[negative_prompt] if negative_prompt else None,
        negative_prompt_2=[negative_prompt] if negative_prompt else None,
    )
    
    # Encode edit concepts as full phrases: e.g. "a boy" + "sad" -> "a sad boy"
    edit_concepts = []
    for edit_prompt in editing_prompts:
        edit_concept_text = _build_edit_concept_text(prompt, edit_prompt)
        (
            edit_embeds,
            _,
            edit_pooled_embeds,
            _,
        ) = pipe.encode_prompt(
            prompt=[edit_concept_text],
            prompt_2=[edit_concept_text],
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        edit_concepts.append({
            'concat': edit_embeds,
            'pooled': edit_pooled_embeds,
            'use_cond_as_negative': True,  # guidance = ε(edit_concept) - ε(origin_prompt)
            'concept_multiplier': 1.0,
        })
    
    cond = {
        'concat': prompt_embeds,
        'pooled': pooled_prompt_embeds,
        'edit_concepts': edit_concepts,
    }
    
    uc = {
        'concat': negative_prompt_embeds,
        'pooled': negative_pooled_prompt_embeds,
    }
    
    return {'cond': cond, 'uc': uc}


if __name__ == "__main__":
    # Example instantiation
    guider = SEGASDXLGuider(
        edit_guidance_scale=5.0,
        edit_threshold=0.95,
        edit_warmup_steps=10,
        edit_momentum_scale=0.3,
        edit_mom_beta=0.6,
        unconditional_guidance_scale=7.5,
    )
    
    print("SEGA SDXL Guider created successfully!")
    print(f"Edit guidance scale: {guider.edit_guidance_scale}")
    print(f"Edit threshold: {guider.edit_threshold}")
    print(f"Warmup steps: {guider.edit_warmup_steps}")
