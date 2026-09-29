import torch
from transformers import GPT2Config
import os
import pandas as pd
import argparse
from model import EmotionInjectionTransformer
from diffusers import StableDiffusionXLPipeline

def emoticrafter(pipe, eit, prompt, a=0, v=0, device="cuda"):
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
    
    # Generate emotional embedding
    out = eit(inputs_embeds=prompt_embeds_ori.to(torch.float32),
              arousal=torch.FloatTensor([[a]]).to(device),
              valence=torch.FloatTensor([[v]]).to(device))
    
    # Return the residual (emotional embedding - original embedding)
    # Note: get_residual.py returns (out[0] - prompt_embeds_ori)
    emotional_residual = out[0] - prompt_embeds_ori
    return emotional_residual

def main():
    parser = argparse.ArgumentParser(description="Generate residuals from music VA CSV")
    parser.add_argument('--csv_path', type=str, default='./music-va/test.csv', help='Path to the CSV file with VA values')
    parser.add_argument('--output_dir', type=str, default='./music-residual', help='Directory to save residuals')
    parser.add_argument('--prompt', type=str, default="forest", help='Prompt to use for generation')
    parser.add_argument('--ckpt_path', type=str, default='./pretrained_model/scale_factor_2.0.pth')
    parser.add_argument('--sdxl_path', type=str, default='../../sdxl_model')
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load Models
    print("Loading models...")
    config = GPT2Config.from_pretrained('./config')
    eit = EmotionInjectionTransformer(config, final_out_type="Linear+LN").to(device)
    eit = torch.nn.DataParallel(eit)
    
    if os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location=device)
        eit.load_state_dict(ckpt)
    else:
        print(f"Warning: Checkpoint not found at {args.ckpt_path}")
        
    eit.eval()
    
    pipe = StableDiffusionXLPipeline.from_pretrained(args.sdxl_path, torch_dtype=torch.float16, 
                                                    use_safetensors=True, variant="fp16")
    pipe.to(device)

    # 2. Read CSV
    print(f"Reading CSV from {args.csv_path}...")
    try:
        # Try reading as tab-separated first, if that fails or looks wrong, try comma
        df = pd.read_csv(args.csv_path, sep='\t')
        if 'valence_raw' not in df.columns:
             df = pd.read_csv(args.csv_path) # Try default comma separator
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if 'valence_raw' not in df.columns or 'arousal_raw' not in df.columns:
        print("Error: CSV must contain 'valence_raw' and 'arousal_raw' columns")
        print(f"Columns found: {df.columns}")
        return

    # 3. Create Output Directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving residuals to {args.output_dir}")

    # 4. Generate Residuals
    print(f"Generating residuals for {len(df)} time steps...")
    
    for index, row in df.iterrows():
        # Use index as time to generate sequence based on CSV length
        time_sec = float(index)
        # Normalize from [1, 9] to [-1, 1]
        valence = (row['valence_raw'] - 5.0) / 4.0
        arousal = (row['arousal_raw'] - 5.0) / 4.0
        
        # Generate residual
        with torch.no_grad():
            residual = emoticrafter(pipe, eit, args.prompt, a=arousal, v=valence, device=device)
        
        # Save residual
        # Filename format: residual_t{time}_v{valence}_a{arousal}.pt
        filename = f"residual_t{time_sec:.1f}.pt"
        save_path = os.path.join(args.output_dir, filename)
        
        torch.save(residual, save_path)
        
        if index % 10 == 0:
            print(f"Processed time {time_sec:.1f}s: V={valence:.2f}, A={arousal:.2f} -> {filename}")

    print("Done!")

if __name__ == "__main__":
    main()
