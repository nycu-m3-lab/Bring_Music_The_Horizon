import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from dataset import DEAMDataset, PMEmoDataset
from mert_model import MERTContinuousModel
import matplotlib.pyplot as plt
import os
import random
from torch.utils.data import Subset
import numpy as np

# --- CONFIG ---
# DEAM Paths
DEAM_AUDIO = "data/DEAM/audio"
DEAM_AR = "data/DEAM/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv"
DEAM_VAL = "data/DEAM/annotations/annotations averaged per song/dynamic (per second annotations)/valence.csv"

# PMEmo Paths
PMEMO_AUDIO = "data/PMEmo/PMEmo2019/chorus"
PMEMO_CSV = "data/PMEmo/PMEmo2019/annotations/dynamic_annotations.csv"

BATCH_SIZE = 1
HEAD_LR = 1e-4
MERT_LR = 1e-6
WEIGHT_DECAY = 1e-5
LOSS_ALPHA = 0.7
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 1e-4
GRAD_CLIP_NORM = 1.0
UNFREEZE_LAST_MERT_LAYER = True
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "results_0624"
FIXED_TEST_SONGS_PATH = "results_0614/test_songs.txt"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- CUSTOM LOSS: Combined MSE and CCC ---
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # pred/target shape: (Batch, Time, 2)
        # We compute CCC independently for Valence (idx 0) and Arousal (idx 1)
        
        ccc_val = self.get_ccc(pred[:, :, 0], target[:, :, 0])
        ccc_aro = self.get_ccc(pred[:, :, 1], target[:, :, 1])
        
        # Loss is 1 - CCC (because we want to maximize CCC)
        ccc_loss = 1.0 - (ccc_val + ccc_aro) / 2.0
        mse_loss = self.mse(pred, target)
        
        return self.alpha * ccc_loss + (1.0 - self.alpha) * mse_loss

    def get_ccc(self, x, y):
        # x, y shape: (Batch, Time) -> Flatten to (Batch * Time)
        
        # Use reshape() to handle memory layout issues safely
        x = x.reshape(-1) 
        y = y.reshape(-1)
        
        mx = torch.mean(x)
        my = torch.mean(y)
        vx = torch.var(x, unbiased=False)
        vy = torch.var(y, unbiased=False)
        cov = torch.mean((x - mx) * (y - my))
        
        eps = 1e-8
        ccc = (2 * cov) / (vx + vy + (mx - my)**2 + eps)
        return ccc

def compute_ccc(x, y):
    x = x.reshape(-1)
    y = y.reshape(-1)

    mx = torch.mean(x)
    my = torch.mean(y)

    vx = torch.var(x, unbiased=False)
    vy = torch.var(y, unbiased=False)

    cov = torch.mean((x - mx) * (y - my))

    eps = 1e-8

    ccc = (2 * cov) / (vx + vy + (mx - my) ** 2 + eps)

    return ccc.item()

def align_preds(preds, labels):
    # Interpolate predictions to match label length
    preds_transposed = preds.permute(0, 2, 1)
    preds_aligned = F.interpolate(preds_transposed, size=labels.shape[1], mode='linear', align_corners=False)
    return preds_aligned.permute(0, 2, 1)

def compute_mse(preds, labels):
    return F.mse_loss(preds, labels).item()

def read_song_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def write_song_list(path, songs):
    with open(path, "w", encoding="utf-8") as f:
        for song in songs:
            f.write(f"{song}\n")

def evaluate(model, loader, loss_fn):

    model.eval()

    total_loss = 0
    total_mse = 0

    val_preds = []
    val_labels = []

    with torch.no_grad():

        for audio, labels in loader:

            audio = audio.to(DEVICE)
            labels = labels.to(DEVICE)

            preds = model(audio)

            preds_aligned = align_preds(preds, labels)

            loss = loss_fn(preds_aligned, labels)

            total_loss += loss.item()
            total_mse += compute_mse(preds_aligned, labels)

            val_preds.append(preds_aligned.cpu())
            val_labels.append(labels.cpu())

    # Concatenate all batches
    val_preds = torch.cat(val_preds, dim=0)
    val_labels = torch.cat(val_labels, dim=0)

    # Compute separate CCC
    valence_ccc = compute_ccc(
        val_preds[:, :, 0],
        val_labels[:, :, 0]
    )

    arousal_ccc = compute_ccc(
        val_preds[:, :, 1],
        val_labels[:, :, 1]
    )

    avg_ccc = (valence_ccc + arousal_ccc) / 2
    valence_mse = compute_mse(val_preds[:, :, 0], val_labels[:, :, 0])
    arousal_mse = compute_mse(val_preds[:, :, 1], val_labels[:, :, 1])

    return (
        total_loss / len(loader),
        total_mse / len(loader),
        valence_ccc,
        arousal_ccc,
        avg_ccc,
        valence_mse,
        arousal_mse,
    )

def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label='Train Combined Loss')
    plt.plot(epochs, history["val_loss"], label='Val Combined Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Lower is Better)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{SAVE_DIR}/loss_history.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["val_ccc"], label='Avg CCC')
    plt.plot(epochs, history["valence_ccc"], label='Valence CCC')
    plt.plot(epochs, history["arousal_ccc"], label='Arousal CCC')
    plt.xlabel('Epoch')
    plt.ylabel('CCC (Higher is Better)')
    plt.title('Validation CCC')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{SAVE_DIR}/ccc_history.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["val_mse"], label='Avg MSE')
    plt.plot(epochs, history["valence_mse"], label='Valence MSE')
    plt.plot(epochs, history["arousal_mse"], label='Arousal MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (Lower is Better)')
    plt.title('Validation MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{SAVE_DIR}/mse_history.png")
    plt.close()

def build_optimizer(model):
    mert_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("mert."):
            mert_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": HEAD_LR, "name": "head"})
    if mert_params:
        param_groups.append({"params": mert_params, "lr": MERT_LR, "name": "mert"})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    head_count = sum(p.numel() for p in head_params)
    mert_count = sum(p.numel() for p in mert_params)
    print(f"Trainable head parameters: {head_count:,} at lr={HEAD_LR:g}")
    print(f"Trainable MERT parameters: {mert_count:,} at lr={MERT_LR:g}")
    print(f"Weight decay: {WEIGHT_DECAY:g}")

    return optimizer

def visualize_test_sample(model, dataset, sample_idx, name):
    model.eval()
    # Get single sample
    audio, labels = dataset[sample_idx]
    
    # Add batch dim
    audio = audio.unsqueeze(0).to(DEVICE)
    labels = labels.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        preds = model(audio)
        preds = align_preds(preds, labels)
        
    # To CPU for plotting
    pred_np = preds.squeeze().cpu().numpy()
    true_np = labels.squeeze().cpu().numpy()
    
    # Plot Valence
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(true_np[:, 0], label='Ground Truth', color='green', linestyle='--')
    plt.plot(pred_np[:, 0], label='Prediction', color='blue')
    plt.title(f"{name} - Valence")
    plt.legend()
    plt.ylim(-1, 1)
    
    # Plot Arousal
    plt.subplot(1, 2, 2)
    plt.plot(true_np[:, 1], label='Ground Truth', color='green', linestyle='--')
    plt.plot(pred_np[:, 1], label='Prediction', color='orange')
    plt.title(f"{name} - Arousal")
    plt.legend()
    plt.ylim(-1, 1)
    
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/test_vis_{name}.png")
    plt.close()

def main():
    print(f"Using device: {DEVICE}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    # --- MODIFIED: Load Both Datasets & Combine ---
    print("--- Loading DEAM ---")
    deam_ds = DEAMDataset(DEAM_AUDIO, DEAM_AR, DEAM_VAL)
    
    print("\n--- Loading PMEmo ---")
    pmemo_ds = PMEmoDataset(PMEMO_AUDIO, PMEMO_CSV)
    
    # Combine them into one big dataset
    full_ds = ConcatDataset([deam_ds, pmemo_ds])
    print(f"\nTotal Combined Samples: {len(full_ds)}")
    # ----------------------------------------------
    
    # -------------------------------------------------
    # SONG-LEVEL SPLIT
    # -------------------------------------------------

    all_indices = list(range(len(full_ds)))

    song_to_indices = {}

    for idx in all_indices:

        # Handle ConcatDataset indexing
        if idx < len(deam_ds):
            sample = deam_ds.samples[idx]
        else:
            sample = pmemo_ds.samples[idx - len(deam_ds)]

        song_id = sample[0]

        if song_id not in song_to_indices:
            song_to_indices[song_id] = []

        song_to_indices[song_id].append(idx)

    # Unique songs
    all_songs = list(song_to_indices.keys())
    random.shuffle(all_songs)

    # Split songs. If FIXED_TEST_SONGS_PATH is set, reuse that held-out test set
    # and split train/val from the remaining songs.
    n_total = len(all_songs)
    n_val = int(0.15 * n_total)

    if FIXED_TEST_SONGS_PATH:
        fixed_test_songs = read_song_list(FIXED_TEST_SONGS_PATH)
        available_songs = set(song_to_indices.keys())
        test_songs = [song for song in fixed_test_songs if song in available_songs]
        missing_test_songs = [song for song in fixed_test_songs if song not in available_songs]
        remaining_songs = [song for song in all_songs if song not in set(test_songs)]

        if missing_test_songs:
            preview = ", ".join(missing_test_songs[:20])
            print(
                f"Warning: {len(missing_test_songs)} fixed test songs were not found "
                f"in this dataset: {preview}"
            )

        print(f"Using fixed test split from: {FIXED_TEST_SONGS_PATH}")
        val_songs = remaining_songs[:n_val]
        train_songs = remaining_songs[n_val:]
    else:
        n_train = int(0.7 * n_total)
        train_songs = all_songs[:n_train]
        val_songs = all_songs[n_train:n_train+n_val]
        test_songs = all_songs[n_train+n_val:]

    # Collect indices
    train_indices = []
    val_indices = []
    test_indices = []

    for s in train_songs:
        train_indices.extend(song_to_indices[s])

    for s in val_songs:
        val_indices.extend(song_to_indices[s])

    for s in test_songs:
        test_indices.extend(song_to_indices[s])

    # Create subsets
    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)
    test_ds = Subset(full_ds, test_indices)

    print(f"Train songs: {len(train_songs)}")
    print(f"Val songs:   {len(val_songs)}")
    print(f"Test songs:  {len(test_songs)}")
    
    # Save the split for rigorous benchmarking later.
    write_song_list(f"{SAVE_DIR}/train_songs.txt", train_songs)
    write_song_list(f"{SAVE_DIR}/val_songs.txt", val_songs)
    write_song_list(f"{SAVE_DIR}/test_songs.txt", test_songs)
        
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    # 3. Initialize Model & CCC Loss
    model = MERTContinuousModel(unfreeze_last_layer=UNFREEZE_LAST_MERT_LAYER).to(DEVICE)
    print(f"Unfreeze last MERT layer: {UNFREEZE_LAST_MERT_LAYER}")
    optimizer = build_optimizer(model)
    loss_fn = CombinedLoss(alpha=LOSS_ALPHA)
    
    # --- OPTIMIZATION 3: SCHEDULER ---
    # Reduces LR if validation loss stops improving for 2 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    # ---------------------------------
    
    best_val_loss = float('inf')
    best_val_ccc = -float('inf')
    epochs_without_improvement = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mse": [],
        "valence_mse": [],
        "arousal_mse": [],
        "val_ccc": [],
        "valence_ccc": [],
        "arousal_ccc": [],
    }
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (audio, labels) in enumerate(train_loader):
            audio, labels = audio.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(audio)
            preds_aligned = align_preds(preds, labels)
            
            loss = loss_fn(preds_aligned, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                GRAD_CLIP_NORM,
            )
            optimizer.step()
            epoch_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"  [Epoch {epoch} Batch {batch_idx}] Loss: {loss.item():.4f}")

        # Validation
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss, avg_val_mse, val_ccc_v, val_ccc_a, val_ccc, val_mse_v, val_mse_a = evaluate(
            model,
            val_loader,
            loss_fn
        )
        
        print(f"    Val CCC Valence: {val_ccc_v:.4f}")
        print(f"    Val CCC Arousal: {val_ccc_a:.4f}")
        print(f"    Avg CCC:         {val_ccc:.4f}")
        
        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_mse"].append(avg_val_mse)
        history["valence_mse"].append(val_mse_v)
        history["arousal_mse"].append(val_mse_a)
        history["val_ccc"].append(val_ccc)
        history["valence_ccc"].append(val_ccc_v)
        history["arousal_ccc"].append(val_ccc_a)
        
        print(f"=== Epoch {epoch} Done ===")
        print(f"    Train Loss: {avg_train_loss:.4f}")
        print(f"    Val Loss:   {avg_val_loss:.4f}")
        print(f"    Val MSE:    {avg_val_mse:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_loss_model.pth")
            print("    [Saved new best loss model]")

        # Save the primary model by CCC because this is the main continuous-affect metric.
        if val_ccc > best_val_ccc + EARLY_STOPPING_MIN_DELTA:
            best_val_ccc = val_ccc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")
            print("    [Saved new best CCC model]")
        else:
            epochs_without_improvement += 1
            print(
                f"    No CCC improvement for {epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE} epochs"
            )

        # Update metric graphs
        plot_history(history)

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"\nEarly stopping: validation CCC did not improve by "
                f"{EARLY_STOPPING_MIN_DELTA:g} for {EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    print("\nTraining Complete. Running Test Visualization...")

    best_model_path = f"{SAVE_DIR}/best_model.pth"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        print(f"Loaded best CCC model for visualization: {best_model_path}")
    
    # 5. Final Test Visualization
    # Pick 3 random samples from the test set
    indices = random.sample(range(len(test_ds)), 3)
    for i, idx in enumerate(indices):
        visualize_test_sample(model, test_ds, idx, f"sample_{i}")
        print(f"Generated plot for test sample {i}")

if __name__ == "__main__":
    main()