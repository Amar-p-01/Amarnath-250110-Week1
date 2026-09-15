"""
Making the dataset loader for the 2018 Data Science Bowl nuclei segmentation task.
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2


def load_image(path: str) -> np.ndarray:
    # Load an image as RGB uint8 array. It can handle grayscale inputs.
    img = Image.open(path)
    img = img.convert("RGB")
    return np.array(img)


def build_combined_mask(masks_dir: str, shape) -> np.ndarray:
    # Together every mask file into one binary mask.
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if not os.path.isdir(masks_dir):
        return mask
    for fname in os.listdir(masks_dir):
        fpath = os.path.join(masks_dir, fname)
        m = np.array(Image.open(fpath).convert("L"))
        mask = np.maximum(mask, (m > 0).astype(np.uint8))
    return mask


class NucleiDataset(Dataset):

    def __init__(self, root_dir: str, img_size: int = 256, train: bool = True,
                 augment: bool = False):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.train = train
        self.augment = augment
        self.sample_ids = sorted(
            [p.name for p in self.root_dir.iterdir() if p.is_dir()]
        )
        if len(self.sample_ids) == 0:
            raise ValueError(f"No sample folders found under {root_dir}")

    def __len__(self):
        return len(self.sample_ids)

    def _image_path(self, sample_dir: Path) -> str:
        img_dir = sample_dir / "images"
        files = list(img_dir.glob("*"))
        if len(files) == 0:
            raise FileNotFoundError(f"No image found in {img_dir}")
        return str(files[0])

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sample_dir = self.root_dir / sample_id
        img_path = self._image_path(sample_dir)
        image = load_image(img_path)
        orig_h, orig_w = image.shape[:2]

        image_r = cv2.resize(image, (self.img_size, self.img_size),
                              interpolation=cv2.INTER_LINEAR)

        if self.train:
            masks_dir = sample_dir / "masks"
            mask = build_combined_mask(str(masks_dir), image.shape)
            mask_r = cv2.resize(mask, (self.img_size, self.img_size),
                                 interpolation=cv2.INTER_NEAREST)

            if self.augment:
                image_r, mask_r = self._augment(image_r, mask_r)

            image_t = torch.from_numpy(image_r / 255.0).permute(2, 0, 1).float()
            mask_t = torch.from_numpy(mask_r).unsqueeze(0).float()
            return image_t, mask_t
        else:
            image_t = torch.from_numpy(image_r / 255.0).permute(2, 0, 1).float()
            return image_t, sample_id, (orig_h, orig_w)

    @staticmethod
    def _augment(image, mask):
        if np.random.rand() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        if np.random.rand() < 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()
        k = np.random.randint(0, 4)
        if k:
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()
        return image, mask
