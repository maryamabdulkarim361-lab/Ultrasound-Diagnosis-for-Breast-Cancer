"""
Dataset loader for the BUSI (Breast Ultrasound Images) dataset.

Expected folder structure (as downloaded from Kaggle):

    data/Dataset_BUSI_with_GT/
        benign/
            benign (1).png
            benign (1)_mask.png   <- segmentation mask, skipped here
            ...
        malignant/
            ...
        normal/
            ...
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = ["normal", "benign", "malignant"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def _collect_samples(data_dir):
    samples = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if "_mask" in fname.lower():
                continue
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            samples.append((os.path.join(class_dir, fname), CLASS_TO_IDX[class_name]))
    if len(samples) == 0:
        raise RuntimeError(
            f"No images found under '{data_dir}'. Check that you downloaded and "
            f"unzipped the BUSI dataset into this path (see README.md)."
        )
    return samples


class BUSIDataset(Dataset):
    """Standard torchvision-transform-based dataset."""

    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.samples = _collect_samples(data_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        image = Image.open(filepath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class BUSIDatasetAlbumentations(Dataset):
    """
    Same dataset, but compatible with an Albumentations `transform` (which
    expects a numpy array via `transform(image=np_array)['image']` and
    returns a tensor directly if the pipeline ends with ToTensorV2()).
    """

    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.samples = _collect_samples(data_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        image = Image.open(filepath).convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label
