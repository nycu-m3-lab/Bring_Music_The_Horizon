import json
import sys
import os
import glob
import shutil
import concurrent.futures
import argparse
import fal_client
import requests
import subprocess
import random
import time
import threading
import hashlib

from import_source import DEFAULT_WORKSPACE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

_gemini_client = None
_api_lock = threading.Lock()

MAX_I2V_FRAMES = 161
MIN_FLF2V_FRAMES = 81

FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "/mnt/M3_Lab/Chiikawa/ffmpeg/ffmpeg"

def _get_gemini_client():
    """Return a Gemini client only when GEMINI_API_KEY is configured."""
    global _gemini_client
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    if _gemini_client is None:
        from google import genai

        _gemini_client = genai.Client()
    return _gemini_client


def _gemini_part_from_file(path: str):
    from google.genai import types

    with open(path, "rb") as f:
        image_bytes = f.read()
    ext = path.rsplit(".", 1)[-1].lower()
    mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
    return types.Part.from_bytes(data=image_bytes, mime_type=mime_map.get(ext, "image/png"))


def polish_prompt(base_prompt: str, arousal: float, keyframe_path: str = None) -> str:
    # 360-Specific Camera Motion: Focuses on moving toward the center to simulate VR forward movement
    # arousal-based speed description (not used now)
    if arousal > 1.5:
        speed_desc = "continuous fast-paced forward motion toward the center of the panorama, as if the viewer is running or sprinting rapidly through the 360-degree environment"
    elif arousal < -1.5:
        speed_desc = "continuous, slow, and leisurely forward motion toward the center of the panorama, as if the viewer is taking a gentle stroll. The forward movement into the center must be constant and never stop"
    else:
        speed_desc = "continuous steady forward motion toward the center of the panorama, as if the viewer is walking forward at a normal pace"
    
    # fallback prompt, if you dont want arousal-based speed description
    # speed_desc = "continuous steady forward motion toward the center of the panorama, as if the viewer is walking forward at a normal pace, the movement into the center must be constant and never stop"
    
    # LLM Instructions tailored for panoramic I2V generation
    text_part = f"""I am generating a 360° panorama Image-to-Video (I2V) animation using a diffusion model. I need you to enhance a base prompt with appropriate ambient dynamics.

Base prompt: "{base_prompt}"

Please format the final prompt exactly like this:
360° panorama video of {base_prompt}, [dynamic], {speed_desc}

Replace [dynamic] with 1-2 short phrases describing subtle, natural ambient movements appropriate for the scene (e.g., if the scene is a forest, use "leaves swaying gently in the breeze"). 
Because this is a 360° panorama, keep the dynamics subtle to avoid distorting the equirectangular projection. Do not introduce dramatic events or rapid view changes.

Output ONLY the final polished prompt string. Do not include any other text, quotes, or explanations."""

    _DYNAMIC_FALLBACK_PROMPT = f"360° panorama video of {base_prompt}, with dynamics appropriate to the scene, {speed_desc}."

    gemini_client = _get_gemini_client()
    if gemini_client is None:
        return _DYNAMIC_FALLBACK_PROMPT

    if keyframe_path is not None:
        contents = [_gemini_part_from_file(keyframe_path), text_part]
    else:
        contents = text_part

    retries = 0
    max_retries = 5
    while True:
        try:
            with _api_lock:
                response = gemini_client.models.generate_content(model="gemini-2.5-flash-lite", contents=contents)
                return response.text.strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                retries += 1
                if retries > max_retries:
                    print(f"[Component 5] Gemini quota exhausted after {retries} retries; using base prompt.")
                    return _DYNAMIC_FALLBACK_PROMPT
                wait_time = min(20, 10 + (2 ** retries))
                print(f"[Component 5] Rate limit hit. Wait {wait_time:.1f}s (Retry {retries})...")
                time.sleep(wait_time)
                continue
            print(f"Error: {e}")
            return _DYNAMIC_FALLBACK_PROMPT

def gen_transition_prompt(start_frame_path: str, end_frame_path: str) -> str:
    """Send the start and end frames to Gemini and get a dolly-in transition prompt.

    Args:
        start_frame_path: Local path to the first frame image (transition start).
        end_frame_path:   Local path to the last frame image (transition end).

    Returns:
        A single prompt string describing the camera-dolly-in transition between
        the two frames, suitable for fal-ai/wan-flf2v.
    """
    instruction = (
        "You are given two frames of a 360 degree panorama video sequence: the FIRST image is the START frame "
        "and the SECOND image is the END frame. Write a single concise prompt (max ~80 "
        "words) for a video diffusion model (wan-flf2v) that smoothly transitions from the "
        "START frame to the END frame using a CAMERA DOLLY IN movement — the camera pushes "
        "forward into the scene, as if the viewer is walking toward the center of the image. "
        "Briefly describe the visible scene content of both frames and any subtle motion, "
        "but always emphasise the dolly-in camera push. Do NOT mention camera pan, lateral "
        "movement, rotation, zoom out, dolly out, or backward motion. Return ONLY the prompt "
        "string — no preamble, no explanation, no quotes."
    )

    _TRANSITION_FALLBACK_PROMPT = "Smooth cinematic transition with the camera steadily dollying in toward the center of the scene, as if the viewer is walking forward, blending the start frame into the end frame while preserving scene content and lighting."

    gemini_client = _get_gemini_client()
    if gemini_client is None:
        return _TRANSITION_FALLBACK_PROMPT

    contents = [
        _gemini_part_from_file(start_frame_path),
        _gemini_part_from_file(end_frame_path),
        instruction,
    ]

    retries = 0
    max_retries = 5
    while True:
        try:
            with _api_lock:
                response = gemini_client.models.generate_content(model="gemini-3-flash-preview", contents=contents,)
                polished = (response.text or "").strip().strip('"').strip("'")
                if not polished:
                    print("[transition] Gemini returned empty prompt, using fallback.")
                    return _TRANSITION_FALLBACK_PROMPT
                print(f"[transition] Gemini prompt: {polished}")
                return polished
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                retries += 1
                if retries > max_retries:
                    print(f"[transition] Gemini quota exhausted after {retries} retries; using fallback prompt.")
                    return _TRANSITION_FALLBACK_PROMPT
                wait_time = min(20, 10 + (2 ** retries))
                print(f"[transition] Rate limit hit. Wait {wait_time:.1f}s (Retry {retries})...")
                time.sleep(wait_time)
                continue
            print(f"Error: {e}")
            return _TRANSITION_FALLBACK_PROMPT

def _on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(log["message"])


def compute_file_hash(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def is_valid_video(path: str, min_size: int = 1024) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > min_size

def _extract_last_frame(video_path: str, output_path: str) -> str:
    """Use ffmpeg to grab the very last frame of *video_path* as a PNG."""
    cmd = [
        FFMPEG, "-y",
        "-sseof", "-0.1",          # seek to ~last 0.1 s
        "-i", video_path,
        "-frames:v", "1",          # grab one frame
        "-update", "1",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"[util] Extracted last frame -> {output_path}")
    return output_path

def process_segment(i, kf_curr, kf_next, kf_curr_url, kf_next_url, dyn_frames, trans_frames, fps, base_prompt, arousal, workspace):
    requested_dyn_frames = dyn_frames
    i2v_gen_frames = min(dyn_frames, MAX_I2V_FRAMES)
    needs_i2v_slowdown = i2v_gen_frames < requested_dyn_frames

    if needs_i2v_slowdown:
        print(
            f"[Component 5] Segment {i}: i2v requested {requested_dyn_frames} frames, "
            f"but Fal max is {MAX_I2V_FRAMES}. Generating {i2v_gen_frames} frames, "
            f"then slowing it to {requested_dyn_frames} frames so transition timing stays fixed."
        )

    # 1. Image-to-Video (Dynamic Generation)
    i2v_path = os.path.join(workspace, f"seg{i:03d}_i2v.mp4")
    if not is_valid_video(i2v_path):
        if os.path.exists(i2v_path):
            print(f"[Component 5] Found invalid/empty I2V file, removing and regenerating: {i2v_path}")
            os.remove(i2v_path)
        print(f"[Component 5] Generating I2V for segment {i}...")
        polished = polish_prompt(base_prompt, arousal, keyframe_path=kf_curr)
        print(f" -> Polished Prompt (Arousal={arousal:.2f}): {polished}")
        
        print(f"[Component 5] Generating video (frames={i2v_gen_frames}, target_frames={requested_dyn_frames}, fps={fps}) ...")
        result = fal_client.subscribe(
            "fal-ai/wan/v2.2-a14b/image-to-video",
            arguments={
                "image_url": kf_curr_url,
                "prompt": polished,
                "num_frames": i2v_gen_frames,
                "frames_per_second": fps,
                "seed": 67,
                "interpolator_model": "none",
                "adjust_fps_for_interpolation": "false"
            },
            with_logs=True,
            on_queue_update=_on_queue_update,
        )
        video_url = result["video"]["url"]
        print(f"[Component 5] Downloading {video_url} -> {i2v_path} ...")
        resp = requests.get(video_url, timeout=300)
        resp.raise_for_status()
        with open(i2v_path, "wb") as f:
            f.write(resp.content)

        if needs_i2v_slowdown:
            raw_path = i2v_path.replace(".mp4", "_raw.mp4")
            os.rename(i2v_path, raw_path)
            cmd = [
                FFMPEG, "-y",
                "-i", raw_path,
                "-filter:v", f"setpts={requested_dyn_frames}/{i2v_gen_frames}*PTS",
                "-r", str(fps),
                "-video_track_timescale", "24000",
                i2v_path,
            ]
            print(f"[Component 5] Slowing I2V {i2v_gen_frames} -> {requested_dyn_frames} frames: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True)
            os.remove(raw_path)
            print(f"[Component 5] Speed-adjusted I2V saved to {i2v_path}")
    else:
        print(f"[Component 5] Skipping I2V for segment {i}, already exists.")

    # 2. First-Last-Frame-to-Video (Transition Generation)
    if trans_frames > 0:
        trans_path = os.path.join(workspace, f"seg{i:03d}_flf2v.mp4")
        if not is_valid_video(trans_path):
            if os.path.exists(trans_path):
                print(f"[Component 5] Found invalid/empty transition file, removing and regenerating: {trans_path}")
                os.remove(trans_path)
            print(f"[Component 5] Generating Transition for segment {i}...")
            last_frame_path = os.path.join(workspace, f"_last_frame_seg{i:03d}.png")
            _extract_last_frame(i2v_path, last_frame_path)
            
            # Upload the dynamically extracted last frame directly here
            last_frame_url = fal_client.upload_file(last_frame_path)
            
            trans_prompt = gen_transition_prompt(last_frame_path, kf_next)
            
            actual_gen_frames = max(trans_frames, MIN_FLF2V_FRAMES)
            needs_speedup = trans_frames < MIN_FLF2V_FRAMES

            if needs_speedup:
                print(f"[Component 5] Requested {trans_frames} frames < minimum {MIN_FLF2V_FRAMES}. Generating {actual_gen_frames} frames, will speed up afterwards.")

            print(f"[Component 5] Generating transition (frames={actual_gen_frames}, fps={fps}) ...")
            result = fal_client.subscribe(
                "fal-ai/wan-flf2v",
                arguments={
                    "prompt": trans_prompt,
                    "negative_prompt": (
                        "pan, panning, camera rotation, lateral movement, zoom out, "
                        "dolly out, move backward, receding camera, pull back, "
                        "shaky camera, jitter, distorted transition."
                    ),
                    "start_image_url": last_frame_url,
                    "end_image_url": kf_next_url,
                    "num_frames": actual_gen_frames,
                    "seed": 1000,
                    "frames_per_second": fps,
                    "cfg": 6,
                },
                with_logs=True,
                on_queue_update=_on_queue_update,
            )
            video_url = result["video"]["url"]
            print(f"[Component 5] Downloading {video_url} -> {trans_path} ...")
            resp = requests.get(video_url, timeout=300)
            resp.raise_for_status()
            with open(trans_path, "wb") as f:
                f.write(resp.content)
            
            if needs_speedup:
                speed_factor = actual_gen_frames / trans_frames
                raw_path = trans_path.replace(".mp4", "_raw.mp4")
                os.rename(trans_path, raw_path)

                cmd = [
                    FFMPEG, "-y",
                    "-i", raw_path,
                    "-filter:v", f"setpts={trans_frames}/{actual_gen_frames}*PTS",
                    "-r", str(fps),
                    "-video_track_timescale", "24000",
                    trans_path,
                ]
                print(f"[Component 5] Speeding up {actual_gen_frames} -> {trans_frames} frames (factor={speed_factor:.3f}x): {' '.join(cmd)}")
                subprocess.run(cmd, check=True, capture_output=True)
                os.remove(raw_path)
                print(f"[Component 5] Speed-adjusted video saved to {trans_path}")
            
            if os.path.exists(last_frame_path):
                os.remove(last_frame_path)
        else:
            print(f"[Component 5] Skipping Transition for segment {i}, already exists.")

def main():
    parser = argparse.ArgumentParser(description="Generate selected I2V and transition clips for a workspace.")
    parser.add_argument("workspace", nargs="?", default=DEFAULT_WORKSPACE)
    parser.add_argument("--start-keyframe", type=int, default=0, help="First keyframe index to test.")
    parser.add_argument("--num-keyframes", type=int, default=None, help="Number of consecutive keyframes to test. Example: 5 keyframes generates 4 segment pairs.")
    parser.add_argument("--start", type=int, default=None, help="Advanced: first segment index to generate, inclusive.")
    parser.add_argument("--end", type=int, default=None, help="Advanced: last segment index to generate, inclusive.")
    parser.add_argument("--segments", default=None, help="Advanced: comma-separated exact segment indices, e.g. 0,3,7.")
    args_cli = parser.parse_args()

    workspace = args_cli.workspace
    with open(os.path.join(workspace, "gen_input.json"), "r") as f:
        gen_input = json.load(f)
    with open(os.path.join(workspace, "args.json"), "r") as f:
        args = json.load(f)

    upload_cache = args.get("keyframe_upload_cache", {})
    if not isinstance(upload_cache, dict):
        upload_cache = {}

    keyframes = sorted(glob.glob(os.path.join(workspace, "keyframe_*.png")))
    num_segments = len(keyframes) - 1

    if args_cli.segments:
        selected_segments = sorted({int(x.strip()) for x in args_cli.segments.split(",") if x.strip()})
    elif args_cli.start is not None or args_cli.end is not None:
        start = args_cli.start if args_cli.start is not None else 0
        end = args_cli.end if args_cli.end is not None else num_segments - 1
        selected_segments = list(range(start, end + 1))
    else:
        start_kf = args_cli.start_keyframe
        if args_cli.num_keyframes is None:
            end_kf = len(keyframes) - 1
        else:
            end_kf = start_kf + args_cli.num_keyframes - 1
        end_kf = min(end_kf, len(keyframes) - 1)
        selected_segments = list(range(start_kf, end_kf))

    selected_segments = [i for i in selected_segments if 0 <= i < num_segments]
    if not selected_segments:
        print("[Component 5] ERROR: no valid segment/keyframe range selected.")
        sys.exit(1)
    
    # Validate that gen_input arrays match the number of keyframes
    expected_dynamic_frames = num_segments
    expected_downbeat_frames = len(keyframes)
    
    actual_dynamic_frames = len(gen_input.get("dynamic_frames", []))
    actual_downbeat_frames = len(gen_input.get("downbeat_frame", []))
    
    if actual_dynamic_frames != expected_dynamic_frames or actual_downbeat_frames != expected_downbeat_frames:
        print(f"[Component 5] ERROR: gen_input.json mismatch!")
        print(f"  Keyframes found: {len(keyframes)} (expects {expected_downbeat_frames} downbeat_frame entries)")
        print(f"  Segments (pairs): {num_segments} (expects {expected_dynamic_frames} dynamic_frames entries)")
        print(f"  gen_input downbeat_frame entries: {actual_downbeat_frames}")
        print(f"  gen_input dynamic_frames entries: {actual_dynamic_frames}")
        print(f"[Component 5] Make sure Component 4 was run and gen_input.json was properly updated.")
        sys.exit(1)
    
    needed_keyframe_indices = set()
    for i in selected_segments:
        needed_keyframe_indices.add(i)
        needed_keyframe_indices.add(i + 1)
    selected_keyframes = [keyframes[i] for i in sorted(needed_keyframe_indices)]

    print(f"[Component 5] Selected segments: {selected_segments}")
    print(f"[Component 5] Selected keyframes: {sorted(needed_keyframe_indices)}")
    print(f"[Component 5] Pre-uploading {len(selected_keyframes)} selected keyframes to Fal.ai to cache URLs...")
    keyframe_urls = {}
    cache_changed = False
    for kf in selected_keyframes:
        rel_path = os.path.relpath(kf, workspace)
        file_hash = compute_file_hash(kf)
        cache_entry = upload_cache.get(rel_path)
        if cache_entry and cache_entry.get("sha256") == file_hash and cache_entry.get("url"):
            print(f" -> Reusing cached URL for {os.path.basename(kf)}")
            keyframe_urls[kf] = cache_entry["url"]
        else:
            print(f" -> Uploading {os.path.basename(kf)}")
            url = fal_client.upload_file(kf)
            keyframe_urls[kf] = url
            upload_cache[rel_path] = {"sha256": file_hash, "url": url}
            cache_changed = True

    if cache_changed:
        args["keyframe_upload_cache"] = upload_cache
        with open(os.path.join(workspace, "args.json"), "w") as f:
            json.dump(args, f, indent=4)
        print(f"[Component 5] Updated args.json with cached upload URLs.")

    print(f"[Component 5] Starting Dynamic & Transition Generation for {len(selected_segments)} selected segments...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(
                process_segment,
                i,
                keyframes[i],
                keyframes[i + 1],
                keyframe_urls[keyframes[i]],
                keyframe_urls.get(keyframes[i + 1]),
                gen_input["dynamic_frames"][i],
                gen_input["transition_frames"][i],
                gen_input["fps"],
                gen_input["prompt"],
                gen_input["downbeat_frame"][i]["arousal"],
                workspace,
            )
            for i in selected_segments
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    print("[Component 5] All segments completed.")

if __name__ == "__main__":
    main()