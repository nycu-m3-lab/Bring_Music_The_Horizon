import os

import torch


def prepare_sdxl_embeddings_for_sega_with_residual(
    pipe,
    prompt: str,
    editing_prompts: list,
    negative_prompt: str = None,
    device: str = "cuda",
    residual_path: str = None,
    residual_multiplier: float = 1.0,
):
    """
    Prepare embeddings for SDXL + SEGA, optionally loading a residual embedding.
    Returns: {'cond': cond, 'uc': uc}
    """
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

    edit_concepts = []

    if residual_path and os.path.exists(residual_path):
        print(f"Loading emotional embedding from {residual_path}...")
        try:
            emotional_embeds = torch.load(residual_path, map_location=device)
            emotional_embeds = emotional_embeds.to(device=device, dtype=prompt_embeds.dtype)

            if emotional_embeds.ndim == 2:
                emotional_embeds = emotional_embeds.unsqueeze(0)

            if emotional_embeds.shape[1:] == prompt_embeds.shape[1:]:
                delta = emotional_embeds
                emotional_embeds = prompt_embeds + delta

                target_norm = prompt_embeds.norm()
                current_norm = emotional_embeds.norm()

                print(f"  Original Prompt Norm: {target_norm:.4f}")
                print(f"  Raw Emotional Embedding Norm: {current_norm:.4f}")

                if abs(current_norm - target_norm) > 1e-3:
                    print(f"  Normalizing emotional embedding from {current_norm:.4f} to {target_norm:.4f}.")
                    emotional_embeds = emotional_embeds * (target_norm / current_norm)

                print(f"  Final Emotional Embedding Norm: {emotional_embeds.norm().item():.4f}")

                edit_concepts.append(
                    {
                        "concat": emotional_embeds,
                        "pooled": pooled_prompt_embeds,
                        "use_cond_as_negative": True,
                        "concept_multiplier": residual_multiplier,
                    }
                )
                print("Successfully added emotional embedding as editing concept.")
            else:
                print("WARNING: Emotional embedding shape mismatch. Skipping.")
        except Exception as e:
            print(f"ERROR loading emotional embedding: {e}")

    for edit_prompt in editing_prompts:
        (edit_embeds, _, edit_pooled_embeds, _) = pipe.encode_prompt(
            prompt=[edit_prompt],
            prompt_2=[edit_prompt],
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        edit_concepts.append({"concat": edit_embeds, "pooled": edit_pooled_embeds})

    cond = {"concat": prompt_embeds, "pooled": pooled_prompt_embeds, "edit_concepts": edit_concepts}
    uc = {"concat": negative_prompt_embeds, "pooled": negative_pooled_prompt_embeds}
    return {"cond": cond, "uc": uc}


def prepare_sdxl_embeddings(pipe, prompt: str, negative_prompt: str, device: str = "cuda"):
    """Plain SDXL prompt-pair embeddings (for non-SEGA paths such as TPoS I2I).

    Returns a 4-tuple:
        (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
    """
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt,
    )
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

