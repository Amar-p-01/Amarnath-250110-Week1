import argparse
import glob
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


def plot_overlapping_histograms(total_hist, cell_hist):
    """
    Plots the overall image pixel histogram and cell-only histogram overlapping
    to help identify a threshold boundary.
    """
    plt.figure(figsize=(12, 6))
    bins = np.arange(256)

    # Plot total histogram (background/all pixels)
    plt.bar(
        bins,
        total_hist,
        width=1.0,
        color="#4A90E2",
        alpha=0.5,
        label="Entire Image Pixels",
        edgecolor="none",
    )

    # Plot cell-only histogram
    plt.bar(
        bins,
        cell_hist,
        width=1.0,
        color="#D0021B",
        alpha=0.6,
        label="Cell Pixels (Mask == 255)",
        edgecolor="none",
    )

    plt.title("Pixel Intensity Distribution: Entire Image vs. Cell Regions (0–255)", fontsize=14)
    plt.xlabel("Pixel Intensity (8-bit)", fontsize=12)
    plt.ylabel("Pixel Count", fontsize=12)
    plt.xlim([0, 255])
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()

    print("\nDisplaying overlapping plot... (Close window to proceed with saving)")
    plt.show()


def compute_overlapping_histograms(root_dir, output_path):
    total_histogram = np.zeros(256, dtype=np.int64)
    cell_histogram = np.zeros(256, dtype=np.int64)

    sample_dirs = [
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]
    sample_dirs.sort()

    print(f"Found {len(sample_dirs)} samples in '{root_dir}'. Processing...")

    valid_samples = 0
    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        sample_uuid = os.path.basename(sample_dir)

        # 1. Locate Image
        img_path = os.path.join(sample_dir, "images", f"{sample_uuid}.png")
        if not os.path.exists(img_path):
            candidates = glob.glob(os.path.join(sample_dir, "images", "*.png"))
            if not candidates:
                continue
            img_path = candidates[0]

        # Load image as 8-bit grayscale
        with Image.open(img_path) as img:
            img_arr = np.array(img.convert("L"), dtype=np.uint8)

        # Update full-image histogram
        total_histogram += np.bincount(img_arr.ravel(), minlength=256)

        # 2. Locate and merge all Masks
        mask_folder = os.path.join(sample_dir, "masks")
        mask_paths = glob.glob(os.path.join(mask_folder, "*.png"))

        if mask_paths:
            # Create a combined boolean mask (True wherever ANY mask is 255 / > 0)
            combined_mask = np.zeros(img_arr.shape, dtype=bool)
            for m_path in mask_paths:
                with Image.open(m_path) as m_img:
                    m_arr = np.array(m_img.convert("L"), dtype=np.uint8)
                    combined_mask |= (m_arr == 255)

            # Extract only the cell pixels using boolean indexing
            cell_pixels = img_arr[combined_mask]
            if cell_pixels.size > 0:
                cell_histogram += np.bincount(cell_pixels, minlength=256)

        valid_samples += 1

    print(f"\nCompleted {valid_samples} samples.")
    print(f"Total Pixels: {total_histogram.sum():,}")
    print(f"Cell Pixels:  {cell_histogram.sum():,} ({cell_histogram.sum() / max(1, total_histogram.sum()) * 100:.2f}%)")

    # Plot before saving
    plot_overlapping_histograms(total_histogram, cell_histogram)

    # 3. Save both histograms
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if output_path.endswith(".npy"):
        # Saves a 2x256 array: row 0 = total, row 1 = cells
        combined_data = np.stack([total_histogram, cell_histogram], axis=0)
        np.save(output_path, combined_data)
    elif output_path.endswith(".csv"):
        stacked = np.column_stack([np.arange(256), total_histogram, cell_histogram])
        np.savetxt(
            output_path,
            stacked,
            fmt="%d",
            delimiter=",",
            header="pixel_value,total_count,cell_count",
            comments="",
        )
    elif output_path.endswith(".json"):
        payload = {
            "intensity": list(range(256)),
            "total_histogram": total_histogram.tolist(),
            "cell_histogram": cell_histogram.tolist(),
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
    else:
        if not output_path.endswith(".npy"):
            output_path += ".npy"
        combined_data = np.stack([total_histogram, cell_histogram], axis=0)
        np.save(output_path, combined_data)

    print(f"Histograms saved to: {output_path}")


import os
import glob
import argparse
from PIL import Image


def compute_average_dimensions(root_dir):
    # Find all sample image files matching the directory structure
    image_paths = glob.glob(os.path.join(root_dir, "*", "images", "*.png"))

    if not image_paths:
        # Fallback recursive search if directory layout differs slightly
        image_paths = [
            os.path.join(root, f)
            for root, _, files in os.walk(root_dir)
            for f in files
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
               and "masks" not in root.split(os.sep)
        ]

    total_images = len(image_paths)
    if total_images == 0:
        print(f"No valid images found in {root_dir}")
        return

    total_width = 0
    total_height = 0
    min_w, max_w = float("inf"), float("-inf")
    min_h, max_h = float("inf"), float("-inf")

    for path in image_paths:
        with Image.open(path) as img:
            # img.size retrieves (width, height) from the header without decoding pixels
            w, h = img.size
            total_width += w
            total_height += h
            min_w, max_w = min(min_w, w), max(max_w, w)
            min_h, max_h = min(min_h, h), max(max_h, h)

    avg_w = total_width / total_images
    avg_h = total_height / total_images

    print(f"Total Images Analyzed: {total_images}")
    print(f"Average Width:         {avg_w:.2f} px (Min: {min_w}, Max: {max_w})")
    print(f"Average Height:        {avg_h:.2f} px (Min: {min_h}, Max: {max_h})")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Calculate overlapping dataset and cell histograms.")
    # parser.add_argument("--dir", type=str, required=True, help="Path to training dataset directory")
    # parser.add_argument("--out", type=str, required=True, help="Path to save (.csv, .npy, or .json)")
    # args = parser.parse_args()
    #
    # compute_overlapping_histograms(args.dir, args.out)

    parser = argparse.ArgumentParser(description="Find average image resolution in a dataset.")
    parser.add_argument("--dir", type=str, required=True, help="Path to root dataset folder")
    args = parser.parse_args()

    compute_average_dimensions(args.dir)