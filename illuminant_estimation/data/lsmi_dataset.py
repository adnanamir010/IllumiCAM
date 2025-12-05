import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob

class LSMIDataset(Dataset):
    def __init__(self, root_dir, cluster_centers, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            cluster_centers (numpy array): 5x3 array of RGB cluster centers.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.cluster_centers = cluster_centers
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True))
        # Note: Actual LSMI structure might vary, assuming .jpg images for now.
        # Ground truth loading logic will need to be adapted based on actual file format (.mat, .png, etc.)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Placeholder for ground truth loading
        # In a real scenario, we would load the mixture maps here
        # For now, returning just the image and a dummy label/mask
        
        return image, img_path

def angular_error(x, y):
    """
    Calculate angular error between two RGB vectors x and y.
    x, y: (3,) or (N, 3) arrays
    Returns error in degrees.
    """
    x = np.array(x)
    y = np.array(y)
    
    # Normalize
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=-1, keepdims=True)
    
    # Avoid division by zero
    x_norm = np.maximum(x_norm, 1e-8)
    y_norm = np.maximum(y_norm, 1e-8)
    
    x = x / x_norm
    y = y / y_norm
    
    # Dot product
    dot = np.sum(x * y, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    
    return np.degrees(np.arccos(dot))

def map_illuminant_to_cluster(illuminant_rgb, cluster_centers):
    """
    Map a single illuminant RGB to the closest cluster center.
    Returns: cluster_index (0-4)
    """
    errors = [angular_error(illuminant_rgb, center) for center in cluster_centers]
    return np.argmin(errors)
