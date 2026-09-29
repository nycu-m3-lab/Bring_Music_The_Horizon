import torch
from transformers import GPT2Config
import os
import sys
import numpy as np
import re
from model import  EmotionInjectionTransformer
from diffusers import StableDiffusionXLPipeline
import torch
import argparse

_lab5_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lab5_2_360emo_gen_module"))
if _lab5_root not in sys.path:
    sys.path.insert(0, _lab5_root)
from emogen.utils.prompts import prompt_filename_stem


def emoticrafter(pipe,eit, prompt,a = 0, v = 0, device = "cuda", seed = 42 ):
    (   prompt_embeds_ori, 
        negative_prompt_embeds,
        pooled_prompt_embeds_ori, 
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=[prompt],
        prompt_2=[prompt],
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
    )
    resolution= int(1024)
    out = eit(inputs_embeds = prompt_embeds_ori.to(torch.float32),arousal=torch.FloatTensor([[a]]).to(device),valence=torch.FloatTensor([[v]]).to(device))
    # Return the complete emotional embedding (original + residual)
    emotional_embedding = out[0] - prompt_embeds_ori
    return emotional_embedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type = str, default="a beautiful landscape with mountains")
    parser.add_argument('--arousal', type = float, default=3.0)
    parser.add_argument('--valence', type = float, default=3.0)
    parser.add_argument('--ckpt_path', type = str, default='/mnt/M3_Lab/Chiikawa/Project/labs/lab4_emoticrafter/checkpoints/prev_best_model.pth')
    parser.add_argument('--sdxl_path', type = str, default='/mnt/M3_Lab/Chiikawa/sdxl_model')
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'grid_3x3', 'grid_5x5','grid_7x7'], help="Generation mode: 'single', 'grid_3x3', or 'grid_5x5'")
    parser.add_argument('--values', type=float, nargs='+', default=[-10, -5, 0, 5, 10], help="List of values for Arousal and Valence")
    args = parser.parse_args()
    
    prompt = args.prompt
    ckpt_path = args.ckpt_path
    sdxl_path = args.sdxl_path
    
    device = 'cuda'

    config = GPT2Config.from_pretrained('./config')
    eit = EmotionInjectionTransformer(config,final_out_type="Linear+LN").to(device)
    ckpt = torch.load(ckpt_path)
    # Handle both DataParallel and non-DataParallel checkpoints
    if list(ckpt.keys())[0].startswith('module.'):
        # Checkpoint was saved with DataParallel, remove "module." prefix
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    eit.load_state_dict(ckpt)
    eit.eval()
    eit.to(device)
    
    pipe = StableDiffusionXLPipeline.from_pretrained(sdxl_path, torch_dtype=torch.float16, 
                                                    use_safetensors=True, variant="fp16")
    pipe.to(device)
    
    os.makedirs("./residuals", exist_ok=True)
    
    # Sanitize + truncate prompt for filename (must match Lab5 scripts that load the .pt)
    sanitized_prompt = prompt_filename_stem(prompt)

    if args.mode == 'single':
        arousal, valence = args.arousal, args.valence
        # Get the emotional prompt embedding (original + EIT residual)
        emotional_prompt_embedding = emoticrafter(pipe, eit, prompt, a=arousal, v=valence, seed=args.seed)
        
        # Save the emotional embedding
        save_path = f"./residuals/{sanitized_prompt}_a{arousal}_v{valence}_seed{args.seed}.pt"
        torch.save(emotional_prompt_embedding, save_path)
        print(f"Saved emotional embedding to {save_path}")
        print(f"Embedding stats: Mean={emotional_prompt_embedding.mean().item():.4f}, Std={emotional_prompt_embedding.std().item():.4f}")

    elif args.mode == 'grid_3x3':
        values = [-3.0, 0.0, 3.0]
        print(f"Generating 3x3 grid residuals for values: {values}")
        for a in values:
            for v in values:
                print(f"Generating residual for A={a}, V={v}...")
                emotional_prompt_embedding = emoticrafter(pipe, eit, prompt, a=a, v=v, seed=args.seed)
                save_path = f"./residuals/{sanitized_prompt}_a{a}_v{v}_seed{args.seed}.pt"
                torch.save(emotional_prompt_embedding, save_path)
                print(f"Saved to {save_path}")

    elif args.mode == 'grid_5x5':
        if not args.values or len(args.values) != 5:
            values = [-10.0, -5.0, 0.0, 5.0, 10.0]
        else:
            values = args.values
        print(f"Generating 5x5 grid residuals for values: {values}")
        for a in values:
            for v in values:
                print(f"Generating residual for A={a}, V={v}...")
                emotional_prompt_embedding = emoticrafter(pipe, eit, prompt, a=a, v=v, seed=args.seed)
                save_path = f"./residuals/{sanitized_prompt}_a{a}_v{v}_seed{args.seed}.pt"
                torch.save(emotional_prompt_embedding, save_path)
                print(f"Saved to {save_path}")

    elif args.mode == 'grid_7x7':
        if not args.values or len(args.values) != 5:
            values = [x for x in range (-3,4)]
        else:
            values = args.values
        print(f"Generating 7x7 grid residuals for values: {values}")
        for a in values:
            for v in values:
                print(f"Generating residual for A={a}, V={v}...")
                emotional_prompt_embedding = emoticrafter(pipe, eit, prompt, a=a, v=v, seed=args.seed)
                save_path = f"./residuals/{sanitized_prompt}_a{a}_v{v}_seed{args.seed}.pt"
                torch.save(emotional_prompt_embedding, save_path)
                print(f"Saved to {save_path}")