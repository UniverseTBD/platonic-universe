import numpy as np
import os

def write_bin(mat, path):
    mat.astype(np.float64).tofile(path)


def plot_sample_galaxies(hf_ds, modes, comp_mode, resize=False, resize_mode="match", n_cols=8):
    """Plot a sample grid of galaxies for visual sanity checking.

    For image modes (jwst, legacysurvey): 2-row grid (HSC top, comp_mode bottom).
    For non-image modes (sdss, desi): 1-row grid (HSC only).

    Saves to figs/test_sample_{comp_mode}.png
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datasets import load_dataset

    from pu.preprocess import flux_to_pil

    NON_IMAGE_MODES = {"sdss", "desi"}
    has_comp_images = comp_mode not in NON_IMAGE_MODES

    ds = load_dataset(hf_ds, split="train", streaming=True)
    samples = list(ds.take(n_cols))

    n_rows = 2 if has_comp_images else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for col, sample in enumerate(samples):
        # HSC row
        hsc_img = flux_to_pil(sample["hsc_image"], "hsc", modes, resize=resize, resize_mode=resize_mode)
        axes[0][col].imshow(hsc_img)
        axes[0][col].axis("off")
        if col == 0:
            axes[0][col].set_title("HSC", fontsize=10)

        # Comparison row
        if has_comp_images:
            comp_img = flux_to_pil(sample[f"{comp_mode}_image"], comp_mode, modes, resize=resize, resize_mode=resize_mode)
            axes[1][col].imshow(comp_img)
            axes[1][col].axis("off")
            if col == 0:
                axes[1][col].set_title(comp_mode.upper(), fontsize=10)

    plt.tight_layout()
    os.makedirs("figs", exist_ok=True)
    out_path = f"figs/test_sample_{comp_mode}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sample galaxy grid to {out_path}")