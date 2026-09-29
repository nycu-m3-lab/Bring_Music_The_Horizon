import torch
import pandas as pd
import librosa
import numpy as np
import os
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor

class DEAMDataset(Dataset):
    """
    Handles DEAM dataset (Wide format, 15.0s offset)
    """
    def __init__(self, audio_dir, arousal_csv, valence_csv, sample_rate=24000, chunk_sec=30):
        self.audio_dir = audio_dir
        self.sr = sample_rate
        self.chunk_sec = chunk_sec
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)

        print(f"Initializing DEAM Dataset...")
        ar_df = pd.read_csv(arousal_csv, dtype={'song_id': str})
        val_df = pd.read_csv(valence_csv, dtype={'song_id': str})

        # Find Common Columns
        ar_cols = set([c for c in ar_df.columns if c.startswith('sample_')])
        val_cols = set([c for c in val_df.columns if c.startswith('sample_')])
        common_cols = list(ar_cols.intersection(val_cols))
        common_cols.sort(key=lambda x: int(x.split('_')[1].replace('ms', '')))
        
        first_ms = int(common_cols[0].split('_')[1].replace('ms',''))
        self.start_offset_sec = first_ms / 1000.0
        
        merged = pd.merge(val_df, ar_df, on='song_id', suffixes=('_val', '_ar'))
        self.samples = []
        self.song_data = {}

        for _, row in merged.iterrows():
            song_id = row['song_id']
            val_values = row[[c + '_val' for c in common_cols]].values.astype(float)
            ar_values = row[[c + '_ar' for c in common_cols]].values.astype(float)
            labels = torch.tensor(np.stack([val_values, ar_values], axis=1), dtype=torch.float)
            self.song_data[song_id] = labels
            
            steps_per_chunk = self.chunk_sec * 2
            
            # DEAM logic: only keep if long enough
            if len(labels) < steps_per_chunk: continue
                
            step_stride = steps_per_chunk // 2 
            for start_idx in range(0, len(labels) - steps_per_chunk + 1, step_stride):
                if torch.isnan(labels[start_idx:start_idx+steps_per_chunk]).any(): continue
                self.samples.append((song_id, start_idx))

        print(f"  - [DEAM] Loaded {len(self.samples)} valid chunks.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        song_id, start_idx = self.samples[idx]
        window_start = self.start_offset_sec + (start_idx * 0.5)
        
        path = os.path.join(self.audio_dir, f"{song_id}.mp3")
        try:
            audio, _ = librosa.load(path, sr=self.sr, offset=window_start, duration=self.chunk_sec)
        except:
            return self.__getitem__((1 + np.random.randint(len(self))) % len(self))
            
        target_len = self.chunk_sec * self.sr
        if len(audio) < target_len: 
            # Pad with zeros if slightly short
            audio = np.pad(audio, (0, target_len - len(audio)), 'constant')
        else: 
            audio = audio[:target_len]
        
        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt")
        labels = self.song_data[song_id][start_idx : start_idx + (self.chunk_sec * 2)]
        return inputs.input_values.squeeze(), labels


class PMEmoDataset(Dataset):
    """
    Handles PMEmo dataset (Long format, 15.5s offset)
    WITH PADDING FOR SHORT SONGS
    """
    def __init__(self, audio_dir, annotation_file, sample_rate=24000, chunk_sec=30):
        self.audio_dir = audio_dir
        self.sr = sample_rate
        self.chunk_sec = chunk_sec
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        
        print(f"Initializing PMEmo Dataset...")
        
        # Load CSV (Long Format)
        df = pd.read_csv(annotation_file, dtype={'musicId': str})
        df = df.rename(columns={'musicId': 'song_id', 'frameTime': 'time', 'Valence(mean)': 'valence', 'Arousal(mean)': 'arousal'})
        
        # PMEmo Offset
        self.start_offset_sec = df['time'].min() # 15.5
        
        self.samples = []
        self.song_data = {}
        
        grouped = df.groupby('song_id')
        print(f"  - Found {len(grouped)} songs in CSV.")
        
        for song_id, group in grouped:
            group = group.sort_values('time')
            val = group['valence'].values
            aro = group['arousal'].values
            labels = torch.tensor(np.stack([val, aro], axis=1), dtype=torch.float)
            self.song_data[song_id] = labels
            
            steps_per_chunk = self.chunk_sec * 2 # Target steps (e.g., 60)
            current_steps = len(labels)
            
            # --- FIX: Handling Short Songs ---
            if current_steps < steps_per_chunk:
                # If song is short (e.g., 16s), we take it anyway starting at 0
                # We will handle the padding in __getitem__
                self.samples.append((song_id, 0)) 
            else:
                # If long enough, use sliding window
                step_stride = steps_per_chunk // 2 
                for start_idx in range(0, current_steps - steps_per_chunk + 1, step_stride):
                    if torch.isnan(labels[start_idx:start_idx+steps_per_chunk]).any(): continue
                    self.samples.append((song_id, start_idx))
                
        print(f"  - [PMEmo] Loaded {len(self.samples)} valid chunks (Short songs included).")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        song_id, start_idx = self.samples[idx]
        
        # PMEmo Start Time
        window_start = self.start_offset_sec + (start_idx * 0.5)
        
        path = os.path.join(self.audio_dir, f"{song_id}.mp3")
        try:
            # Load as much as we can up to chunk_sec
            audio, _ = librosa.load(path, sr=self.sr, offset=window_start, duration=self.chunk_sec)
        except:
             return self.__getitem__((1 + np.random.randint(len(self))) % len(self))

        target_len = self.chunk_sec * self.sr
        
        # --- AUDIO PADDING ---
        if len(audio) < target_len:
            # Pad with Zeros (Silence)
            padding = target_len - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
            
        else:
            audio = audio[:target_len]
            
        inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt")
        
        # --- LABEL PADDING ---
        target_steps = self.chunk_sec * 2 # 60 steps
        
        # Grab available labels
        # If song is short, this slice will be shorter than target_steps
        raw_labels = self.song_data[song_id][start_idx : start_idx + target_steps]
        
        if len(raw_labels) < target_steps:
            # Pad labels with the LAST known value (Repeat end emotion)
            # This is better than 0 for emotion continuity
            missing = target_steps - len(raw_labels)
            last_val = raw_labels[-1].unsqueeze(0) # Shape (1, 2)
            padding = last_val.repeat(missing, 1)  # Shape (missing, 2)
            labels = torch.cat([raw_labels, padding], dim=0)
        else:
            labels = raw_labels
            
        return inputs.input_values.squeeze(), labels