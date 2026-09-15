import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split

# Custom project imports
from architecture import UNET, UNETUpsampleBilinear
from utils import JaccardLoss
from utils import SegmentationDataset, train_segmentation


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet for Binary Cell Segmentation")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the training data directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation set split ratio (0.0 to 1.0)")
    parser.add_argument("--checkpoint", type=str, default="best_unet.pth", help="Path to save best weights")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Dataset and Train/Val Split
    full_dataset = SegmentationDataset(root_dir=args.data_dir, is_train=True)
    total_samples = len(full_dataset)

    val_size = int(total_samples * args.val_split)
    train_size = total_samples - val_size

    # Deterministic split for reproducible validation sets
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Total samples: {total_samples} | Train: {train_size} | Validation: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    ) if val_size > 0 else None

    # 3. Model instantiation (1 input channel -> grayscale; 1 output channel -> binary mask logits)
    model = UNETUpsampleBilinear()

    # 4. Optimization & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = JaccardLoss(smooth=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    # 5. Kick off training
    train_segmentation(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=args.epochs,
        device=device,
        val_loader=val_loader,
        scheduler=scheduler,
        checkpoint_path=args.checkpoint
    )


if __name__ == "__main__":
    main()