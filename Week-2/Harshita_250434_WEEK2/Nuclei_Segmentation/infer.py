"""
Run inference on stage2_test_final and write a Kaggle-format submission.csv.
"""

import argparse
import csv
import cv2
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import NucleiDataset
from model import UNet
from rle import rle_encode


def split_instances(binary_mask: np.ndarray, min_distance: int = 4):
    """Standard distance-transform watershed, much safer for single-class U-Net binary masks."""
    if binary_mask.sum() == 0:
        return np.zeros_like(binary_mask, dtype=np.int32)

    distance = ndi.distance_transform_edt(binary_mask)
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary_mask)
    
    peak_mask = np.zeros_like(distance, dtype=bool)
    if len(coords) > 0:
        peak_mask[tuple(coords.T)] = True
    markers = ndi.label(peak_mask)[0]

    if markers.max() == 0:
        return ndi.label(binary_mask)[0]

    labels = watershed(-distance, markers, mask=binary_mask)
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="stage2_test_final folder")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.45) # Balanced threshold
    ap.add_argument("--min_distance", type=int, default=4,
                     help="min separation between watershed peaks (px)")
    ap.add_argument("--min_object_size", type=int, default=10,
                     help="drop instance predictions smaller than this many px")
    ap.add_argument("--out", default="submission_stage2.csv")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    ds = NucleiDataset(args.data_dir, img_size=args.img_size, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    model = UNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    rows = [("ImageId", "EncodedPixels")]
    n_images = 0
    n_instances = 0

    with torch.no_grad():
        for image, sample_id, (orig_h, orig_w) in tqdm(loader, desc="Processing Stage 2"):
            sample_id = sample_id[0]
            orig_h, orig_w = int(orig_h), int(orig_w)
            image = image.to(device)

            # TTA: Test-Time Augmentation
            logits_orig = model(image)
            probs_orig = torch.sigmoid(logits_orig)[0, 0].cpu().numpy()

            image_h = torch.flip(image, [3])
            logits_h = model(image_h)
            probs_h = np.flip(torch.sigmoid(logits_h)[0, 0].cpu().numpy(), axis=1)

            image_v = torch.flip(image, [2])
            logits_v = model(image_v)
            probs_v = np.flip(torch.sigmoid(logits_v)[0, 0].cpu().numpy(), axis=0)

            # Average predictions to smooth boundaries
            probs_avg = (probs_orig + probs_h + probs_v) / 3.0

            probs_full = cv2.resize(probs_avg, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            binary = (probs_full > args.threshold).astype(np.uint8)

            # Safer watershed algorithm
            labels = split_instances(binary, min_distance=args.min_distance)
            
            n_images += 1
            found_any = False
            
            for label_id in range(1, labels.max() + 1):
                inst_mask = (labels == label_id).astype(np.uint8)
                if inst_mask.sum() < args.min_object_size:
                    continue
                rle = rle_encode(inst_mask)
                if rle is None:
                    continue
                rows.append((sample_id, rle))
                found_any = True
                n_instances += 1

            if not found_any:
                rows.append((sample_id, ""))

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Wrote {args.out}: {n_images} images, {n_instances} nucleus instances.")

if __name__ == "__main__":
    main()