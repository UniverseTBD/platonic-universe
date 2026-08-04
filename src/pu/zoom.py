import cv2
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from scipy.ndimage import zoom
from skimage.filters import threshold_otsu
from skimage.measure import label


def otsu_threshold(img):
    """Apply Otsu's thresholding with Gaussian blur"""
    if len(img.shape) == 3:
        img_gray = np.mean(img, axis=2)
    else:
        img_gray = img

    # Apply Gaussian blur
    img_blurred = cv2.GaussianBlur(img_gray.astype(np.float32), (41, 41), 10)

    # Apply Otsu's thresholding
    threshold_val = threshold_otsu(img_blurred)
    mask = img_blurred > threshold_val

    return mask, threshold_val


def find_center_blob_info(mask):
    """Find the size and extent of the contiguous blob at the center of the image"""
    center_h, center_w = mask.shape[0] // 2, mask.shape[1] // 2

    # Label connected components
    labeled = label(mask.astype(int))

    # Find which component the center pixel belongs to
    center_label = labeled[center_h, center_w]

    if center_label == 0:
        return 0, None  # No blob at center

    # Create mask for only the center blob
    center_blob_mask = labeled == center_label

    # Count pixels in the center blob
    center_blob_size = np.sum(center_blob_mask)

    # Find extent of only the center blob
    rows, cols = np.where(center_blob_mask)
    if len(rows) > 0:
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        extent = (min_row, max_row, min_col, max_col)
    else:
        extent = None

    return center_blob_size, extent


def resize_galaxy_to_fit(img, padding_ratio=0.1, target_size=96, force_extent=None):
    """
    Estimate galaxy size and resize centered galaxy to 96x96

    Args:
        img: Grayscale numpy array (2D), galaxy assumed to be centered
        padding_ratio: Ratio of padding around the galaxy (0.1 = 10% padding)
        target_size: Output size (default 96x96)
        force_extent: Force clip to these boundaries
            (min_row, max_row, min_col, max_col = force_extent)

    Returns:
        resized_img: Galaxy resized to target_size x target_size
        galaxy_size: Number of pixels in the contiguous center blob
    """
    # Get binary mask using Otsu's method
    mask, threshold = otsu_threshold(img)

    if force_extent is None:
        # Count only the contiguous blob at the center
        galaxy_size, extent = find_center_blob_info(mask)

        if galaxy_size == 0 or extent is None:
            return img[
                img.shape[0] // 2 - target_size // 2 : img.shape[0] // 2
                + target_size // 2,
                img.shape[1] // 2 - target_size // 2 : img.shape[1] // 2
                + target_size // 2,
            ]
    else:
        galaxy_size = 0
        extent = force_extent

    min_row, max_row, min_col, max_col = extent

    # Calculate galaxy dimensions
    galaxy_length = max([max_row - min_row + 1, max_col - min_col + 1])
    pad = int(galaxy_length * padding_ratio)
    # Calculate crop bounds (centered)
    center_h, center_w = img.shape[0] // 2, img.shape[1] // 2
    crop = galaxy_length + 2 * pad

    # Crop from center
    start_row = max(0, center_h - crop // 2)
    end_row = min(img.shape[0], center_h + crop // 2)
    start_col = max(0, center_w - crop // 2)
    end_col = min(img.shape[1], center_w + crop // 2)

    galaxy_crop = img[start_row:end_row, start_col:end_col]

    max_dim = max(galaxy_crop.shape[0], galaxy_crop.shape[1])
    if max_dim <= 0:
        # catch zero division error
        return img[
            img.shape[0] // 2 - target_size // 2 : img.shape[0] // 2 + target_size // 2,
            img.shape[1] // 2 - target_size // 2 : img.shape[1] // 2 + target_size // 2,
        ], 0

    # Resize to target size (96x96)
    if len(galaxy_crop.shape) == 3:
        # RGB image: apply zoom to each channel
        zoom_factor = target_size / max_dim
        resized = zoom(galaxy_crop, (zoom_factor, zoom_factor, 1), order=1)
    else:
        # Grayscale image
        zoom_factor = target_size / max_dim
        resized = zoom(galaxy_crop, zoom_factor, order=1)

    return resized


# Example usage
if __name__ == "__main__":
    ds = iter(load_dataset("<anon>/galaxies", split="train", streaming=True))
    for ii in range(10):
        img = np.array(next(ds)["image"])[:, :, 0]

        # Apply the galaxy resize function
        resized_img, galaxy_size = resize_galaxy_to_fit(img, padding_ratio=0.15)
        f, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(resized_img)
        axs[1].imshow(img)
        plt.show()

        print(f"Galaxy size: {galaxy_size} pixels")
        print(f"Output shape: {resized_img.shape}")
