import os
import pandas as pd
import numpy as np
from skimage import exposure, transform
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class CBISDDSMDataset(Dataset):
    """
    A simplified dataloader for the CBIS-DDSM dataset.
    Assumes images are pre-cropped and saved in a structured directory.
    """
    def __init__(self, csv_file, img_dir, transform=None, stack_slices=10):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.stack_slices = stack_slices  # Number of slices to stack for 3D volume

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Apply preprocessing: Resize, Normalize, Histogram Equalization
        image = self.preprocess_image(image)

        # Label: 0=Normal, 1=Benign, 2=Malignant
        label = int(self.annotations.iloc[idx, 1])

        # For 3D models, we need to simulate a volume by stacking the same slice.
        # A real implementation would load adjacent slices.
        volume = np.stack([image] * self.stack_slices, axis=0)
        volume = torch.FloatTensor(volume).unsqueeze(0)  # Adds channel dim: (1, Depth, H, W)

        if self.transform:
            volume = self.transform(volume)

        return volume, label

    def preprocess_image(self, image):
        """Applies preprocessing to a single 2D mammogram image."""
        # Resize
        image = transform.resize(image, (256, 256), anti_aliasing=True)
        # Normalize to [0, 1]
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        # Histogram Equalization
        image = exposure.equalize_hist(image)
        # Convert back to 0-255 range for potential later transforms
        image = (image * 255).astype(np.uint8)
        return image

def get_dataloaders(csv_train, csv_val, img_dir, batch_size=4, stack_slices=10):
    """Creates training and validation dataloaders."""
    train_dataset = CBISDDSMDataset(csv_train, img_dir, stack_slices=stack_slices)
    val_dataset = CBISDDSMDataset(csv_val, img_dir, stack_slices=stack_slices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader
