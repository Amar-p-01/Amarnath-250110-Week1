import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from torch import nn

BRIGHTNESS_FLOOR = 9
TARGET_SIZE = (384, 384)


class SegmentationDataset(Dataset):
    """
    Loads samples from:
      Training:
        <root_dir>/<sample_uuid>/images/<sample_uuid>.png
        <root_dir>/<sample_uuid>/masks/*.png (Multiple masks)
      Testing:
        <root_dir>/<sample_uuid>/images/<sample_uuid>.png
    """

    def __init__(
            self,
            root_dir,
            is_train=True,
            transform=None,
            brightness_floor=BRIGHTNESS_FLOOR,
            target_size=TARGET_SIZE,
    ):
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform
        self.brightness_floor = brightness_floor
        self.target_size = target_size
        self.normalized_threshold = brightness_floor / 255.0

        # Discover all sample UUID folders
        self.sample_dirs = [
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        self.sample_dirs.sort()

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        sample_uuid = os.path.basename(sample_dir)

        # 1. Load Grayscale/BW Image
        img_path = os.path.join(sample_dir, "images", f"{sample_uuid}.png")
        if not os.path.exists(img_path):
            candidates = glob.glob(os.path.join(sample_dir, "images", "*.png"))
            if not candidates:
                raise FileNotFoundError(f"No image found in {sample_dir}/images")
            img_path = candidates[0]

        image = Image.open(img_path).convert("L")
        image = TF.to_tensor(image)  # [1, H, W] in range [0.0, 1.0]

        # 2. Dark Noise Floor Clipping
        if self.brightness_floor > 0:
            image[image <= self.normalized_threshold] = 0.0

        # 3. Resize Image (Bilinear)
        image = TF.resize(
            image,
            size=self.target_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        # Return early for test set
        if not self.is_train:
            if self.transform:
                image = self.transform(image)
            return image, sample_uuid

        # 4. Merge all masks via cumulative sum + clamp to maintain a binary mask
        mask_folder = os.path.join(sample_dir, "masks")
        mask_paths = glob.glob(os.path.join(mask_folder, "*.png"))

        # Initialize accumulated mask tensor
        merged_mask = torch.zeros((1, *self.target_size), dtype=torch.float32)

        if mask_paths:
            for m_path in mask_paths:
                m_img = Image.open(m_path).convert("L")
                m_tensor = TF.to_tensor(m_img)  # [1, H, W]

                # Resize each individual mask using nearest-neighbor to retain edge fidelity
                m_tensor = TF.resize(
                    m_tensor,
                    size=self.target_size,
                    interpolation=InterpolationMode.NEAREST,
                )

                # Accumulate (sum) overlapping / adjacent cell regions
                merged_mask += (m_tensor > 0.5).float()

            # Clamp sum to 1.0 so overlapping regions stay binary (0.0 or 1.0)
            merged_mask = torch.clamp(merged_mask, min=0.0, max=1.0)

        # 5. Synchronized Transforms (e.g. geometric flips/rotations)
        if self.transform:
            image, merged_mask = self.transform(image, merged_mask)

        return image, merged_mask


def train_segmentation(
        model,
        train_loader,
        optimizer,
        loss_fn,
        num_epochs,
        device,
        val_loader=None,
        scheduler=None,
        checkpoint_path="best_model.pth",
):
    """
    Standard training and evaluation loop targeting single-channel merged masks.
    """
    model.to(device)
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}] Train", leave=False)
        for images, targets in pbar:
            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)  # [B, 1, 384, 384]

            optimizer.zero_grad()

            # Forward pass -> outputs raw logits of shape [B, 1, 384, 384]
            preds = model(images)
            loss = loss_fn(preds, targets)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        epoch_val_loss = None
        if val_loader is not None:
            model.eval()
            running_val_loss = 0.0

            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device, dtype=torch.float32)
                    targets = targets.to(device, dtype=torch.float32)

                    preds = model(images)
                    loss = loss_fn(preds, targets)
                    running_val_loss += loss.item() * images.size(0)

            epoch_val_loss = running_val_loss / len(val_loader.dataset)

            # Checkpoint on best validation score
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(model.state_dict(), checkpoint_path)
                print(f"[*] Best model saved at epoch {epoch} (Val Loss: {epoch_val_loss:.4f})")

        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss if epoch_val_loss is not None else epoch_train_loss)
            else:
                scheduler.step()

        val_msg = f" | Val Loss: {epoch_val_loss:.4f}" if epoch_val_loss is not None else ""
        print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}{val_msg}")

    print("Training finished.")


class JaccardLoss(nn.Module):
    """
    Soft Jaccard / IoU Loss for binary segmentation.
    Expects raw unnormalized model logits.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert raw logits to probabilities [0.0, 1.0]
        probs = torch.sigmoid(logits)

        # Flatten batch and spatial dimensions for intersection computation
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        total = (probs_flat + targets_flat).sum(dim=1)
        union = total - intersection

        # Soft IoU calculation per batch item
        iou = (intersection + self.smooth) / (union + self.smooth)

        # Loss is 1 - IoU, averaged across the batch
        return (1.0 - iou).mean()