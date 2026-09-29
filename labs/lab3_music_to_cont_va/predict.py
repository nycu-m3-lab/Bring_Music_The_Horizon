import torch
import torch.nn.functional as F
import librosa
import numpy as np
import matplotlib.pyplot as plt
from transformers import Wav2Vec2FeatureExtractor
from mert_model import MERTContinuousModel
import os

# --- CONFIG ---
# Path to the model trained by the new train.py
SAVE_DIR = "results_deam_only"
MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pth")
# Change this to the song you want to test!
TEST_SONG = "../../songs/michael-jackson---stranger-in-moscow-official-video.mp3"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_song(audio_path):
    print(f"--- Analyzing: {audio_path} ---")
    
    # 1. Initialize Model
    model = MERTContinuousModel().to(DEVICE)
    
    # Load weights (map_location ensures it works even if trained on GPU and running on CPU)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Did you run train.py?")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 2. Process Audio
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    
    # Load audio (resample to 24k for MERT)
    # We load up to 600 seconds to prevent running out of GPU memory on massive songs
    MAX_DURATION = 600
    audio, sr = librosa.load(audio_path, sr=24000, duration=MAX_DURATION)
    
    print(f"Audio loaded: {len(audio)/sr:.2f} seconds")

    inputs = processor(audio, sampling_rate=24000, return_tensors="pt")
    input_values = inputs.input_values.to(DEVICE)
    
    # 3. Inference
    with torch.no_grad():
        preds = model(input_values) # Shape: (1, Time_Steps, 2)
        
    # Move to CPU/Numpy for plotting
    preds = preds.squeeze().cpu().numpy()
    
    # 4. Generate Dynamic Graph
    # Create a time axis (e.g., 0.0s, 0.1s, 0.2s...) matching the number of predictions
    time_axis = np.linspace(0, len(audio)/24000, len(preds))
    
    plt.figure(figsize=(12, 6))
    
    # Plot Valence (Positivity)
    plt.plot(time_axis, preds[:, 0], label='Valence (Positivity)', color='blue', linewidth=2)
    
    # Plot Arousal (Energy)
    plt.plot(time_axis, preds[:, 1], label='Arousal (Energy)', color='orange', linewidth=2)
    
    # Formatting
    plt.title(f"Dynamic Emotion Tracking: {os.path.basename(audio_path)}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Value (-1 to 1)")
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5) # Center line
    plt.ylim(-1.1, 1.1)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks at 60-second intervals
    max_time = len(audio)/24000
    plt.xticks(np.arange(0, max_time + 60, 60))
    
    # Save
    output_path = os.path.join(SAVE_DIR, f"prediction_of_{os.path.basename(audio_path).replace('.mp3', '')}.png")
    plt.savefig(output_path)
    print(f"Graph saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    predict_song(TEST_SONG)