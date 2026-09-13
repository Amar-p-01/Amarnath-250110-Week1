import argparse
import glob
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

# Import model architecture
from architecture import UNET

BRIGHTNESS_FLOOR = 9
TARGET_SIZE = (384, 384)


def load_and_preprocess_image(img_path, target_size=TARGET_SIZE, brightness_floor=BRIGHTNESS_FLOOR):
    """Loads, clips dark noise, and resizes image to tensor format."""
    raw_img = Image.open(img_path).convert("L")
    img_tensor = TF.to_tensor(raw_img)  # [1, H, W] in range [0.0, 1.0]

    # Apply brightness floor
    if brightness_floor > 0:
        normalized_threshold = brightness_floor / 255.0
        img_tensor[img_tensor <= normalized_threshold] = 0.0

    # Resize to model input dimensions (Bilinear)
    img_tensor = TF.resize(
        img_tensor,
        size=target_size,
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return img_tensor


def load_and_merge_masks(mask_folder, target_size=TARGET_SIZE):
    """Discovers, resizes (nearest-neighbor), sums, and clamps all masks into one binary mask."""
    mask_paths = glob.glob(os.path.join(mask_folder, "*.png"))
    if not mask_paths:
        return None

    merged_mask = torch.zeros((1, *target_size), dtype=torch.float32)
    for m_path in mask_paths:
        m_img = Image.open(m_path).convert("L")
        m_tensor = TF.to_tensor(m_img)

        # Nearest-neighbor to preserve crisp binary edges
        m_tensor = TF.resize(
            m_tensor,
            size=target_size,
            interpolation=InterpolationMode.NEAREST,
        )
        merged_mask += (m_tensor > 0.5).float()

    merged_mask = torch.clamp(merged_mask, min=0.0, max=1.0)
    return merged_mask.squeeze().numpy()


def run_inference(model, img_tensor, device, threshold=0.5):
    """Runs forward pass and converts logits to a thresholded binary numpy array."""
    input_batch = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_batch)
        probs = torch.sigmoid(logits)
        # pred_mask = (probs > threshold).float()
        pred_mask = probs.float()

    return pred_mask.squeeze().cpu().numpy()


def visualize(img_tensor, pred_mask, gt_mask, sample_uuid):
    """Renders visual inspection comparison."""
    img_np = img_tensor.squeeze().cpu().numpy()

    if gt_mask is not None:
        # 4-panel comparison: Image, Ground Truth, Prediction, Dual Overlay
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(img_np, cmap="gray")
        axes[0].set_title(f"Input ({TARGET_SIZE[0]}x{TARGET_SIZE[1]})")
        axes[0].axis("off")

        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth (Merged)")
        axes[1].axis("off")

        axes[2].imshow(pred_mask, cmap="gray")
        axes[2].set_title("Predicted Mask")
        axes[2].axis("off")

        # Overlay: Green = GT, Red = Pred, Yellow (Red+Green) = True Positive Overlap
        overlay = np.stack([img_np * 0.7, img_np * 0.7, img_np * 0.7], axis=-1)
        overlay[gt_mask > 0.5, 1] = 1.0    # Green channel for Ground Truth
        overlay[pred_mask > 0.5, 0] = 1.0  # Red channel for Predictions
        # Intersections automatically become [1.0, 1.0, 0.0] -> Yellow

        axes[3].imshow(overlay)
        axes[3].set_title("Overlay (G: GT, R: Pred, Y: Overlap)")
        axes[3].axis("off")

    else:
        # Fallback 3-panel for test directories without a masks folder
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_np, cmap="gray")
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(pred_mask, cmap="gray")
        axes[1].set_title("Predicted Mask")
        axes[1].axis("off")

        overlay = np.stack([img_np, img_np, img_np], axis=-1)
        overlay[pred_mask > 0.5] = [1.0, 0.2, 0.2]

        axes[2].imshow(overlay)
        axes[2].set_title("Prediction Overlay")
        axes[2].axis("off")

    plt.suptitle(f"Sample UUID: {sample_uuid}", fontsize=13)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Inference and Ground Truth comparison")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--checkpoint", type=str, default="best_unet.pth", help="Path to saved weights (.pth)")
    parser.add_argument("--threshold", type=float, default=0.95, help="Classification probability threshold")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Select random sample directory
    sample_dirs = [
        os.path.join(args.data_dir, d)
        for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    ]
    if not sample_dirs:
        raise ValueError(f"No valid sample folders in {args.data_dir}")

    sample_dir = random.choice(sample_dirs)
    sample_uuid = os.path.basename(sample_dir)

    # 2. Locate image
    img_path = os.path.join(sample_dir, "images", f"{sample_uuid}.png")
    if not os.path.exists(img_path):
        candidates = glob.glob(os.path.join(sample_dir, "images", "*.png"))
        if not candidates:
            raise FileNotFoundError(f"No image found in {sample_dir}/images")
        img_path = candidates[0]

    print(f"Sample Selected: {sample_uuid}")

    # 3. Load & preprocess Image and GT Mask (if present)
    img_tensor = load_and_preprocess_image(img_path)

    mask_folder = os.path.join(sample_dir, "masks")
    gt_mask = None
    if os.path.exists(mask_folder):
        gt_mask = load_and_merge_masks(mask_folder)

    # 4. Load Model and Predict
    model = UNET()
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    pred_mask = run_inference(model, img_tensor, device, threshold=args.threshold)
    print(pred_mask)
    for i in range(pred_mask.shape[0]):
        for j in range(pred_mask.shape[1]):
            print(pred_mask[i, j])

    # 5. Plot
    visualize(img_tensor, pred_mask, gt_mask, sample_uuid)


if __name__ == "__main__":
    main()