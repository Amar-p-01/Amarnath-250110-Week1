"""
Run-length encoding utilities matching the DSB2018 submission format.

Kaggle's expected encoding: pixels are numbered from top to bottom, then
left to right (i.e. column-major / Fortran order), 1-indexed. The encoding
is pairs of (start, length).
"""

import numpy as np


def rle_encode(binary_mask: np.ndarray):
    """
    binary_mask: 2D array (H, W) of 0/1 for a SINGLE nucleus instance.
    Returns the RLE string, or None if the mask is empty.
    """
    pixels = binary_mask.T.flatten()  # column-major order per Kaggle spec
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if len(runs) == 0:
        return None
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(rle_str: str, shape):
    """Inverse of rle_encode — mainly useful for sanity-checking a submission."""
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(shape, order="F")
