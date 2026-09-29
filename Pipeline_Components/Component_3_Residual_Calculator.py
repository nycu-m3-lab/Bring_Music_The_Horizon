import json
import os
import sys
import torch

from import_source import DEFAULT_WORKSPACE, check_triton_version, EmotiCrafter

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_COMPONENT_DIR, ".."))
_DEFAULT_EIT_MODEL = os.path.join(_PROJECT_ROOT, "big_files", "emoti_best_model.pth")
_DEFAULT_SDXL_MODEL = os.path.join(_PROJECT_ROOT, "big_files", "sdxl_model")
check_triton_version()

def main():
    workspace = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WORKSPACE
    with open(os.path.join(workspace, "args.json"), "r") as f:
        args = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eit_model = args.get("eit_model") or _DEFAULT_EIT_MODEL
    if not os.path.exists(eit_model) and os.path.exists(_DEFAULT_EIT_MODEL):
        print(f"[Component 3] EIT model not found at {eit_model}; using {_DEFAULT_EIT_MODEL}")
        eit_model = _DEFAULT_EIT_MODEL
    sdxl_path = args.get("sdxl_path") or _DEFAULT_SDXL_MODEL
    if not os.path.exists(sdxl_path) and os.path.exists(_DEFAULT_SDXL_MODEL):
        print(f"[Component 3] SDXL path not found at {sdxl_path}; using {_DEFAULT_SDXL_MODEL}")
        sdxl_path = _DEFAULT_SDXL_MODEL
    emoti = EmotiCrafter(sdxl_path, eit_model, device)

    va_tuples = torch.load(os.path.join(workspace, "va_tuples.pt")).to(device)
    base_prompt = args["prompt"]
    base_seed = args["seed"]

    print("[Component 3] Extracting text embeddings & injecting emotion...")
    (prompt_embeds_ori, _, _, _) = emoti.pipe.encode_prompt(
        prompt=[base_prompt], prompt_2=[base_prompt], device=device, num_images_per_prompt=1, do_classifier_free_guidance=True
    )
    
    residuals_list = []
    torch.Generator(device=device).manual_seed(base_seed)  # Set seed once for all keyframes to maintain consistency
    for i in range(va_tuples.shape[0]):
        # Adjusting the seed sequentially for each section as requested
        # torch.Generator(device=device).manual_seed(base_seed + i)
        # we aint doing that anymore
        v = va_tuples[i, 0].item()
        a = va_tuples[i, 1].item()
        v_tensor = torch.FloatTensor([[v]]).to(device)
        a_tensor = torch.FloatTensor([[a]]).to(device)
        
        out = emoti.eit(inputs_embeds=prompt_embeds_ori.to(torch.float32), arousal=a_tensor, valence=v_tensor)
        residuals_list.append(out[0] - prompt_embeds_ori)

    raw_residuals = torch.cat(residuals_list, dim=0)
    torch.save(raw_residuals, os.path.join(workspace, "raw_residuals.pt"))

    # Create pipeline metadata map
    interval_sec = args["four_bar_sec"]
    frame_interval = int(round(interval_sec * args["fps"]))
    n_kf = va_tuples.shape[0]
    
    # Sync max_keyframes in args.json with actual VA tuple count
    args["max_keyframes"] = n_kf
    with open(os.path.join(workspace, "args.json"), "w") as f:
        json.dump(args, f, indent=4)
    
    dynamic_frames = []
    transition_frames = []
    for i in range(n_kf - 1):
        dyn = int(round(frame_interval * 0.75))
        dynamic_frames.append(dyn)
        transition_frames.append(frame_interval - dyn)

    downbeat_frame = []
    for i in range(n_kf):
        downbeat_frame.append({
            "frame": int(frame_interval * i),
            "start_sec": args["start_sec"] + (interval_sec * i),
            "valence": float(va_tuples[i, 0].item()),
            "arousal": float(va_tuples[i, 1].item())
        })
        
    args.update({"dynamic_frames": dynamic_frames, "transition_frames": transition_frames, "downbeat_frame": downbeat_frame})
    with open(os.path.join(workspace, "gen_input.json"), "w") as f:
        json.dump(args, f, indent=4)
    print(f"[Component 3] gen_input.json created with dynamic/transition frame info and downbeat metadata.")
    
if __name__ == "__main__":
    main()