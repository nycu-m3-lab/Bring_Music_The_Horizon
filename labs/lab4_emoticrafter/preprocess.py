import csv
import json
import random
import torch
import argparse
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdxl_path', type = str, default='/mnt/M3_Lab/Chiikawa/sdxl_model')
    parser.add_argument('--csv_path', type = str, default = './data/prompts.csv', help='Path to CSV file as data source')
    parser.add_argument('--json_path', type = str, default = '/mnt/M3_Lab/Chiikawa/Project/labs/lab7_gemini_gen_dataset/1000and5.json', help='Path to JSON file as data source')
    parser.add_argument('--data_cache_path', type = str, default = "./data/data-cache.pt")
    args = parser.parse_args()
    
    # Load data from JSON or CSV
    if args.json_path and os.path.exists(args.json_path):
        with open(args.json_path, 'r') as f:
            if isinstance(f.read(), str):
                f.seek(0)
                json_data = json.load(f)
            else:
                f.seek(0)
                json_data = json.load(f)
        
        # Handle both single object and list of objects
        if isinstance(json_data, dict):
            data = [json_data]
        else:
            data = json_data
    else:
        with open(args.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
    device = 'cuda'
    index=0
    sdxl_path = args.sdxl_path
    pipe = StableDiffusionXLPipeline.from_pretrained(sdxl_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to(device)
        
    res_list = []
    for item in tqdm(data):
        # Handle both CSV and JSON formats
        if isinstance(item, dict):
            # Determine the field names based on data source
            arousal_key = 'arousal' if 'arousal' in item else 'Arousal'
            valence_key = 'valence' if 'valence' in item else 'Valence'
            neutral_key = 'neutral_prompt' if 'neutral_prompt' in item else 'Neutral_Prompt'
            emotional_key = 'emotional_prompt' if 'emotional_prompt' in item else 'Emotional_Prompt'
            
            # Skip if required fields are missing
            if not item.get(arousal_key) or not item.get(valence_key):
                continue
            
            neural_prompt = item.get(neutral_key, '')
            arousal = float(item.get(arousal_key))
            valence = float(item.get(valence_key))
            emotional_prompt = item.get(emotional_key, '')
        else:
            continue

        with torch.no_grad():
            (   prompt_embeds_ori, 
                negative_prompt_embeds,
                pooled_prompt_embeds_ori, 
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt=[neural_prompt ],
                prompt_2=[neural_prompt ],
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
            
            (  prompt_embeds, 
                negative_prompt_embeds,
                pooled_prompt_embeds_ori, 
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt=[emotional_prompt],
                prompt_2=[emotional_prompt],
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
                    
        neutral_prompt_feature = prompt_embeds_ori[0]  
            
        emotional_prompt_feature = prompt_embeds[0]
        res =   { 
                'neutral_prompt_feature': neutral_prompt_feature.detach().cpu().to(torch.float16),
                'arousal': torch.tensor([arousal], dtype=torch.float),
                'valence': torch.tensor([valence], dtype=torch.float),
                'emotional_prompt_feature': emotional_prompt_feature.detach().cpu().to(torch.float16)
                }
        index+=1
        res_list.append(res)
    torch.save(res_list,args.data_cache_path)