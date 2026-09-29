import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dataset import DEAMDataset, PMEmoDataset
from mert_model import MERTContinuousModel


DEAM_AUDIO = "data/DEAM/audio"
DEAM_AR = "data/DEAM/annotations/annotations averaged per song/dynamic (per second annotations)/arousal.csv"
DEAM_VAL = "data/DEAM/annotations/annotations averaged per song/dynamic (per second annotations)/valence.csv"

PMEMO_AUDIO = "data/PMEmo/PMEmo2019/chorus"
PMEMO_CSV = "data/PMEmo/PMEmo2019/annotations/dynamic_annotations.csv"


class TestSongDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        source_name, dataset, sample_idx, song_id = self.entries[idx]
        audio, labels = dataset[sample_idx]
        return audio, labels, source_name, song_id


def align_preds(preds, labels):
    if preds.shape[1] == labels.shape[1]:
        return preds

    preds = preds.permute(0, 2, 1)
    preds = F.interpolate(preds, size=labels.shape[1], mode="linear", align_corners=False)
    return preds.permute(0, 2, 1)


def ccc_score(pred, target, eps=1e-8):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(target)
    pred = pred[mask]
    target = target[mask]

    if pred.size == 0:
        return float("nan")

    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_var = pred.var()
    target_var = target.var()
    covariance = ((pred - pred_mean) * (target - target_mean)).mean()
    denominator = pred_var + target_var + (pred_mean - target_mean) ** 2
    return float((2.0 * covariance) / (denominator + eps))


def regression_metrics(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(target)
    diff = pred[mask] - target[mask]

    if diff.size == 0:
        return {
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
        }

    mse = float(np.mean(diff ** 2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(diff))),
    }


def summarize_metrics(preds, targets):
    metrics = {}
    names = ["valence", "arousal"]

    for dim, name in enumerate(names):
        dim_metrics = regression_metrics(preds[:, dim], targets[:, dim])
        metrics[f"{name}_mse"] = dim_metrics["mse"]
        metrics[f"{name}_rmse"] = dim_metrics["rmse"]
        metrics[f"{name}_mae"] = dim_metrics["mae"]
        metrics[f"{name}_ccc"] = ccc_score(preds[:, dim], targets[:, dim])

    all_metrics = regression_metrics(preds, targets)
    metrics["mse"] = all_metrics["mse"]
    metrics["rmse"] = all_metrics["rmse"]
    metrics["mae"] = all_metrics["mae"]
    metrics["ccc"] = float(np.nanmean([metrics["valence_ccc"], metrics["arousal_ccc"]]))
    return metrics


def read_test_songs(path):
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def add_matching_entries(entries, source_name, dataset, test_songs):
    found = defaultdict(int)
    for sample_idx, sample in enumerate(dataset.samples):
        song_id = str(sample[0])
        if song_id in test_songs:
            entries.append((source_name, dataset, sample_idx, song_id))
            found[song_id] += 1
    return found


def load_test_dataset(test_songs_path):
    test_songs = read_test_songs(test_songs_path)

    print("--- Loading DEAM ---")
    deam_ds = DEAMDataset(DEAM_AUDIO, DEAM_AR, DEAM_VAL)

    print("\n--- Loading PMEmo ---")
    pmemo_ds = PMEmoDataset(PMEMO_AUDIO, PMEMO_CSV)

    entries = []
    deam_found = add_matching_entries(entries, "DEAM", deam_ds, test_songs)
    pmemo_found = add_matching_entries(entries, "PMEmo", pmemo_ds, test_songs)

    matched = set(deam_found) | set(pmemo_found)
    missing = sorted(test_songs - matched, key=lambda x: int(x) if x.isdigit() else x)

    print(f"\nTest songs listed: {len(test_songs)}")
    print(f"Matched test songs: {len(matched)}")
    print(f"Test chunks: {len(entries)}")
    print(f"  DEAM chunks: {sum(deam_found.values())}")
    print(f"  PMEmo chunks: {sum(pmemo_found.values())}")

    if missing:
        preview = ", ".join(missing[:20])
        print(f"Warning: {len(missing)} test songs had no matching chunks: {preview}")

    if not entries:
        raise RuntimeError("No evaluation chunks matched the test song list.")

    return TestSongDataset(entries)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    per_song = defaultdict(lambda: {"preds": [], "targets": [], "sources": set(), "chunks": 0})

    with torch.no_grad():
        for audio, labels, sources, song_ids in loader:
            audio = audio.to(device)
            labels = labels.to(device)

            preds = model(audio)
            preds = align_preds(preds, labels)

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_preds.append(preds_np.reshape(-1, preds_np.shape[-1]))
            all_targets.append(labels_np.reshape(-1, labels_np.shape[-1]))

            for batch_idx, song_id in enumerate(song_ids):
                per_song[song_id]["preds"].append(preds_np[batch_idx])
                per_song[song_id]["targets"].append(labels_np[batch_idx])
                per_song[song_id]["sources"].add(sources[batch_idx])
                per_song[song_id]["chunks"] += 1

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    overall = summarize_metrics(all_preds, all_targets)
    rows = []

    for song_id, values in per_song.items():
        song_preds = np.concatenate(values["preds"], axis=0)
        song_targets = np.concatenate(values["targets"], axis=0)
        row = summarize_metrics(song_preds, song_targets)
        row["song_id"] = song_id
        row["source"] = "+".join(sorted(values["sources"]))
        row["chunks"] = values["chunks"]
        row["frames"] = int(song_targets.shape[0])
        rows.append(row)

    rows.sort(key=lambda row: int(row["song_id"]) if row["song_id"].isdigit() else row["song_id"])
    return overall, rows


def save_per_song_csv(rows, path):
    if not rows:
        return

    fieldnames = [
        "song_id",
        "source",
        "chunks",
        "frames",
        "ccc",
        "valence_ccc",
        "arousal_ccc",
        "mse",
        "valence_mse",
        "arousal_mse",
        "rmse",
        "valence_rmse",
        "arousal_rmse",
        "mae",
        "valence_mae",
        "arousal_mae",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def print_metric_block(title, metrics):
    print(f"\n{title}")
    print(f"  Avg CCC:     {metrics['ccc']:.4f}")
    print(f"  Valence CCC: {metrics['valence_ccc']:.4f}")
    print(f"  Arousal CCC: {metrics['arousal_ccc']:.4f}")
    print(f"  MSE:         {metrics['mse']:.6f}")
    print(f"  RMSE:        {metrics['rmse']:.6f}")
    print(f"  MAE:         {metrics['mae']:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained continuous valence/arousal model on saved test songs."
    )
    parser.add_argument("--results-dir", default="results_0624")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--test-songs", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--per-song-csv", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = args.checkpoint or os.path.join(args.results_dir, "best_model.pth")
    test_songs = args.test_songs or os.path.join(args.results_dir, "test_songs.txt")
    per_song_csv = args.per_song_csv or os.path.join(args.results_dir, "test_metrics_per_song.csv")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not os.path.exists(test_songs):
        raise FileNotFoundError(f"Test song list not found: {test_songs}")

    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Test songs: {test_songs}")

    dataset = load_test_dataset(test_songs)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = MERTContinuousModel(unfreeze_last_layer=True).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    overall, per_song_rows = evaluate(model, loader, device)
    print_metric_block("Overall Test Metrics", overall)

    save_per_song_csv(per_song_rows, per_song_csv)
    print(f"\nSaved per-song metrics to: {per_song_csv}")


if __name__ == "__main__":
    main()
