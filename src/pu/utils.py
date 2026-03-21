import numpy as np
import os

def write_bin(mat, path):
    mat.astype(np.float64).tofile(path)


BAND_INFO = {
    "hsc": {"indices": [0, 1, 3], "names": ["g", "r", "z"]},
    "jwst": {"indices": [0, 4, 6], "names": ["f090w", "f277w", "f444w"]},
    "legacysurvey": {"indices": [0, 1, 3], "names": ["g", "r", "z"]},
}


def plot_sample_galaxies(hf_ds, modes, comp_mode, resize=True, resize_mode="match", n_cols=8):
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

    # --- Per-band diagnostic plots ---
    if has_comp_images:
        _plot_bands(samples, "hsc", modes, resize, resize_mode, n_cols)
        _plot_bands(samples, comp_mode, modes, resize, resize_mode, n_cols)


def _plot_bands(samples, mode, modes, resize, resize_mode, n_cols):
    """Plot each band separately for a given mode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pu.preprocess import flux_to_pil
    from pu.zoom import resize_galaxy_to_fit

    info = BAND_INFO.get(mode)
    if info is None:
        return

    band_indices = info["indices"]
    band_names = info["names"]
    n_bands = len(band_names)

    fig, axes = plt.subplots(n_bands, n_cols, figsize=(2 * n_cols, 2 * n_bands))

    for col, sample in enumerate(samples):
        arr = np.asarray(sample[f"{mode}_image"]["flux"], np.float32)
        if arr.ndim != 3:
            continue

        for row, (idx, bname) in enumerate(zip(band_indices, band_names)):
            chan = arr[idx]

            # Apply same resize as flux_to_pil by wrapping in 3-channel then taking one
            if resize and mode in ("hsc", "legacysurvey"):
                dummy = np.stack([chan, chan, chan], axis=-1)
                if mode == "hsc":
                    extent = (68, 92, 68, 92) if resize_mode == "match" else None
                else:
                    extent = (72, 88, 72, 88) if resize_mode == "match" else None
                dummy = resize_galaxy_to_fit(
                    dummy, target_size=96,
                    force_extent=extent,
                )
                chan = dummy[..., 0]

            axes[row][col].imshow(chan, cmap="gray")
            axes[row][col].axis("off")
            if col == 0:
                axes[row][col].set_title(f"{mode.upper()} {bname}", fontsize=10)

    plt.tight_layout()
    os.makedirs("figs", exist_ok=True)
    out_path = f"figs/test_bands_{mode}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-band diagnostic plot to {out_path}")