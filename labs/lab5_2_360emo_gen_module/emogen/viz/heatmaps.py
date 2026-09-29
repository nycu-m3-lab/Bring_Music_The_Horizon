import os

import torch.nn.functional as F


def save_heatmap_images(heatmaps, output_dir, target_size=(1024, 1024), selected_steps=None):
    """
    Save heatmaps (edit_range and psi) as images.
    edit_range = where edit_threshold is applied (1 = guidance active, 0 = inactive).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found, skipping heatmap save.")
        return

    os.makedirs(output_dir, exist_ok=True)
    if not heatmaps:
        print("No heatmaps to save.")
        return

    for hm in heatmaps:
        step = hm["step"]
        if selected_steps is not None and step not in selected_steps:
            continue
        for key in ["edit_range", "psi", "psi_with_momentum"]:
            if key not in hm:
                continue
            heatmap = hm[key][0]
            heatmap_up = (
                F.interpolate(
                    heatmap.unsqueeze(0).unsqueeze(0),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .numpy()
            )
            if heatmap_up.max() > heatmap_up.min():
                heatmap_up = (heatmap_up - heatmap_up.min()) / (heatmap_up.max() - heatmap_up.min())
            plt.figure(figsize=(10, 10 * target_size[0] / target_size[1]))
            plt.imshow(heatmap_up, cmap="hot", interpolation="bilinear")
            plt.colorbar(label="Edit range" if key == "edit_range" else "|ψ|")
            plt.title(f"Step {step} - {key}")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"heatmap_step{step:03d}_{key}.png"),
                dpi=100,
                bbox_inches="tight",
                pad_inches=0.1,
            )
            plt.close()
    print(f"Heatmaps saved to {output_dir}")


def create_heatmap_grid(heatmaps, output_path, target_size=(256, 256), max_cols=5, key="edit_range"):
    """Create a grid of heatmaps (edit_range by default) over steps."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not heatmaps:
        return
    n = len(heatmaps)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols

    aspect = target_size[1] / target_size[0]
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows * aspect))

    if rows == 1 and cols == 1:
        import numpy as np

        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, hm in enumerate(heatmaps):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        if key not in hm:
            ax.axis("off")
            continue
        heatmap = hm[key][0]
        heatmap_up = (
            F.interpolate(
                heatmap.unsqueeze(0).unsqueeze(0),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .numpy()
        )
        if heatmap_up.max() > heatmap_up.min():
            heatmap_up = (heatmap_up - heatmap_up.min()) / (heatmap_up.max() - heatmap_up.min())
        ax.imshow(heatmap_up, cmap="hot", interpolation="bilinear")
        ax.set_title(f"Step {hm['step']}")
        ax.axis("off")

    for idx in range(n, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Heatmap grid saved to {output_path}")


def save_tpos_heatmap_grid(
    heatmap_store,
    heatmap_dir: str,
    target_size=(800, 1600),
):
    """Render delta_sparse and tpos_guidance grids for the TPoS I2I sampler.

    ``heatmap_store`` is a list of ``(step_index, t, delta_map, guidance_map)``
    tuples collected during sampling, where the maps are 2-D float CPU tensors.
    Two PNGs are written under ``heatmap_dir``: ``heatmap_grid_delta_sparse.png``
    and ``heatmap_grid_tpos_guidance.png``.
    """
    if not heatmap_store:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found, skipping TPoS heatmap save.")
        return

    import numpy as np

    os.makedirs(heatmap_dir, exist_ok=True)

    def _render(map_idx: int, name: str):
        n = len(heatmap_store)
        cols = 5
        rows = (n + cols - 1) // cols

        H, W = target_size
        fig_w = 4 * cols
        fig_h = fig_w * (H / W) * (rows / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten() if isinstance(axes, np.ndarray) else np.array([axes])

        for idx, ax in enumerate(axes):
            if idx < n:
                entry = heatmap_store[idx]
                step_i = entry[0]
                data = entry[map_idx]
                heatmap_up = (
                    F.interpolate(
                        data.unsqueeze(0).unsqueeze(0),
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze()
                    .numpy()
                )
                if heatmap_up.max() > heatmap_up.min():
                    heatmap_up = (heatmap_up - heatmap_up.min()) / (heatmap_up.max() - heatmap_up.min())

                ax.imshow(heatmap_up, cmap="hot", interpolation="bilinear", aspect="equal")
                ax.set_title(f"Step {step_i}")
                ax.axis("off")
            else:
                ax.axis("off")

        plt.tight_layout()
        out_path = os.path.join(heatmap_dir, f"heatmap_grid_{name}.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"Saved {name} heatmap grid to {out_path}")

    _render(map_idx=2, name="delta_sparse")
    _render(map_idx=3, name="tpos_guidance")

