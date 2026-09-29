import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from import_source import DEFAULT_WORKSPACE, VAExtractor

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_COMPONENT_DIR, ".."))
_DEFAULT_VA_MODEL = os.path.join(_PROJECT_ROOT, "big_files", "music2VA_best_model.pth")


def save_va_curve_plot(va_tensor: torch.Tensor, hz: float, out_path: str) -> None:
    """Save valence / arousal vs time as a two-panel PNG."""
    va_np = va_tensor.detach().cpu().numpy()
    t = np.arange(va_np.shape[0], dtype=np.float64) / float(hz)
    v = va_np[:, 0]
    a = va_np[:, 1]

    fig, (ax_v, ax_a) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 6), constrained_layout=True
    )
    ax_v.plot(t, v, color="#2ecc71", linewidth=0.9)
    ax_v.set_ylabel("Valence (v)")
    ax_v.grid(True, alpha=0.35)
    ax_a.plot(t, a, color="#e74c3c", linewidth=0.9)
    ax_a.set_ylabel("Arousal (a)")
    ax_a.set_xlabel("Time (s)")
    ax_a.grid(True, alpha=0.35)
    fig.suptitle("Valence / Arousal (scaled) vs time")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    workspace = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WORKSPACE
    with open(os.path.join(workspace, "args.json"), "r") as f:
        args = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    va_model = args.get("va_model") or _DEFAULT_VA_MODEL
    if not os.path.exists(va_model) and os.path.exists(_DEFAULT_VA_MODEL):
        print(f"[Component 2] VA model not found at {va_model}; using {_DEFAULT_VA_MODEL}")
        va_model = _DEFAULT_VA_MODEL
    va = VAExtractor(va_model, device)
    
    print(f"[Component 2] Extracting VA tuples on {device} from the full audio file...")
    va_tuples = va.extract_va_tuples(args["audio_path"], args["hz"])

    # Basic Auto-Scaling logic mimicking MVPPipeline
    v_list = va_tuples[:, 0].tolist()
    a_list = va_tuples[:, 1].tolist()
    candidates = []
    if max(v_list) > 0: candidates.append(3.0 / max(v_list))
    if min(v_list) < 0: candidates.append(-3.0 / min(v_list))
    if max(a_list) > 0: candidates.append(3.0 / max(a_list))
    if min(a_list) < 0: candidates.append(-3.0 / min(a_list))
    
    scale = min([s for s in candidates if s > 0], default=1.0)
    scaled_va_tuples = va_tuples * scale

    torch.save(scaled_va_tuples, os.path.join(workspace, "va_tuples.pt"))
    print(f"[Component 2] Saved VA tensors scaled by {scale:.4f}.")

    plot_path = os.path.join(workspace, "va_curves.png")
    save_va_curve_plot(scaled_va_tuples, float(args["hz"]), plot_path)
    print(f"[Component 2] Saved VA curve plot: {plot_path}")


if __name__ == "__main__":
    main()