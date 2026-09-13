import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from architecture import UNET
from utils import SegmentationDataset


def compute_hard_jaccard(pred_binary: torch.Tensor, target_binary: torch.Tensor, smooth: float = 1e-6) -> float:
    """Computes dataset-level or sample-level binary IoU (Jaccard Index)."""
    intersection = (pred_binary * target_binary).sum().item()
    total = pred_binary.sum().item() + target_binary.sum().item()
    union = total - intersection
    return (intersection + smooth) / (union + smooth)


def sweep_thresholds(model, dataloader, device, thresholds):
    model.eval()

    all_probs = []
    all_targets = []

    print("Running forward passes to collect model predictions...")
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Inference"):
            images = images.to(device, dtype=torch.float32)

            # Predict logits -> Sigmoid probabilities
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Keep tensors on CPU to preserve GPU memory
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate across all batches: [N, 1, H, W]
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    best_threshold = None
    best_iou = -1.0
    best_loss = float("inf")

    results = []

    print("\nEvaluating thresholds...")
    for t in thresholds:
        pred_binary = (all_probs >= t).float()

        # Calculate hard IoU (Jaccard Score)
        iou = compute_hard_jaccard(pred_binary, all_targets)
        jaccard_loss = 1.0 - iou

        results.append((t, iou, jaccard_loss))

        if iou > best_iou:
            best_iou = iou
            best_loss = jaccard_loss
            best_threshold = t

    return results, best_threshold, best_iou, best_loss


def main():
    parser = argparse.ArgumentParser(description="Sweep decision threshold for optimal Jaccard Loss")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--checkpoint", type=str, default="best_unet.pth", help="Path to saved model weights")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--start", type=float, default=0.85, help="Sweep starting threshold")
    parser.add_argument("--end", type=float, default=0.1, help="Sweep ending threshold")
    parser.add_argument("--step", type=float, default=0.025, help="Step size for threshold sweep")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset & Loader
    dataset = SegmentationDataset(root_dir=args.data_dir, is_train=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Loaded dataset with {len(dataset)} samples.")

    # 2. Model
    model = UNET()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    # 3. Generate threshold candidates
    thresholds = np.arange(args.start, args.end + 1e-5, args.step)

    # 4. Run sweep
    results, best_t, best_iou, best_loss = sweep_thresholds(model, dataloader, device, thresholds)

    # 5. Display table
    print("\n" + "=" * 45)
    print(f"{'Threshold':<12}{'Hard IoU (Jaccard)':<20}{'Jaccard Loss':<15}")
    print("-" * 45)
    for t, iou, loss in results:
        marker = " <-- BEST" if np.isclose(t, best_t) else ""
        print(f"{t:<12.3f}{iou:<20.4f}{loss:<15.4f}{marker}")
    print("=" * 45)

    print(f"\nBest Threshold:    {best_t:.3f}")
    print(f"Best Jaccard Loss: {best_loss:.4f} (IoU: {best_iou:.4f})")


if __name__ == "__main__":
    main()