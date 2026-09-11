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


def load_and_preprocess(img_path, target_size=TARGET_SIZE, brightness_floor=BRIGHTNESS_FLOOR):
    """Loads, clips dark noise, and resizes single image to tensor format."""
    raw_img = Image.open(img_path).convert("L")
    img_tensor = TF.to_tensor(raw_img)  # [1, H, W] in [0.0, 1.0]

    # Apply brightness floor
    if brightness_floor > 0:
        normalized_threshold = brightness_floor / 255.0
        img_tensor[img_tensor <= normalized_threshold] = 0.0

    # Resize to model input dimensions
    img_tensor = TF.resize(
        img_tensor,
        size=target_size,
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return img_tensor


def run_inference(model, img_tensor, device, threshold=0.5):
    """Runs forward pass and converts logits to a thresholded binary mask."""
    # Add batch dimension: [1, 1, H, W]
    input_batch = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_batch)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > threshold).float()

    # Squeeze batch & channel dimensions -> [H, W] numpy array
    return pred_mask.squeeze().cpu().numpy()


def visualize_prediction(image_tensor, pred_mask, sample_uuid):
    """Displays original image, predicted mask, and a red boundary overlay."""
    img_np = image_tensor.squeeze().cpu().numpy()

    # Build RGB overlay (red tint on masked regions)
    overlay = np.stack([img_np, img_np, img_np], axis=-1)
    overlay[pred_mask > 0.5] = [1.0, 0.2, 0.2]  # Highlight detected cells in light red

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title(f"Input Image ({TARGET_SIZE[0]}x{TARGET_SIZE[1]})")
    axes[0].axis("off")

    axes[1].imshow(pred_mask, cmap="gray")
    axes[1].set_title("Predicted Mask (Binary)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Mask Overlay on Image")
    axes[2].axis("off")

    plt.suptitle(f"Sample UUID: {sample_uuid}", fontsize=13)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run inference on a random image from dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to test or train directory")
    parser.add_argument("--checkpoint", type=str, default="best_unet.pth", help="Path to saved .pth weights")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for mask")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # 1. Discover sample folders
    sample_dirs = [
        os.path.join(args.data_dir, d)
        for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    ]
    if not sample_dirs:
        raise ValueError(f"No sample directories found in {args.data_dir}")

    # Pick a random sample
    sample_dir = random.choice(sample_dirs)
    sample_uuid = os.path.basename(sample_dir)

    img_path = os.path.join(sample_dir, "images", f"{sample_uuid}.png")
    if not os.path.exists(img_path):
        candidates = glob.glob(os.path.join(sample_dir, "images", "*.png"))
        if not candidates:
            raise FileNotFoundError(f"No image found in {sample_dir}/images")
        img_path = candidates[0]

    print(f"Selected Sample: {sample_uuid}")
    print(f"Image Path: {img_path}")

    # 2. Load trained model
    model = UNET()
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # 3. Preprocess & predict
    img_tensor = load_and_preprocess(img_path)
    pred_mask = run_inference(model, img_tensor, device, threshold=args.threshold)

    # 4. Display results
    visualize_prediction(img_tensor, pred_mask, sample_uuid)


if __name__ == "__main__":
    main()