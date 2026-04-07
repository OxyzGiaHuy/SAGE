import os
import random
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
import cv2
from sklearn.model_selection import train_test_split
import logging
Image.MAX_IMAGE_PIXELS = None 


def get_transformations(img_size):
    """
    Get data augmentation transformations for training and validation
    """

    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5, 
                           border_mode=cv2.BORDER_CONSTANT), 
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transforms = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transforms

class UniversalMedicalDataset(Dataset):
    """
    A generic dataset loader for medical segmentation tasks.
    Assumes a flat directory structure or simple subfolders.
    Automatically splits data into train/val/test if not pre-split.
    """
    def __init__(self, root_dir, split='train', transform=None, 
                 img_ext='.png', mask_suffix='_mask', 
                 split_ratios=(0.7, 0.15, 0.15), seed=42, image_size=512):
        """
        Args:
            root_dir: Path to the dataset root.
            split: 'train', 'val', or 'test'.
            transform: Albumentations transforms (optional, will use default if None).
            img_ext: Extension of images (e.g., '.png', '.jpg').
            mask_suffix: Suffix for mask files (e.g., '_mask', '_seg').
            split_ratios: Tuple of (train, val, test) ratios.
            seed: Random seed for reproducibility.
            image_size: Target image size for transformations.
        """
        self.root_dir = root_dir
        self.split = split
        self.mask_suffix = mask_suffix
        self.img_ext = img_ext
        self.image_size = image_size
        
        # 1. Collect all image files
        self.all_images = []
        self.all_masks = []
        self.case_names = []
        
        # Walk through directory to find images
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(img_ext) and mask_suffix not in file:
                    img_path = os.path.join(root, file)
                    
                    # Construct expected mask path
                    # Assumes mask is in the same folder or a parallel 'masks' folder
                    # Strategy 1: Same folder, suffix check
                    mask_name = file.replace(img_ext, mask_suffix + img_ext)
                    mask_path = os.path.join(root, mask_name)
                    
                    # Strategy 2: Check if mask exists
                    if os.path.exists(mask_path):
                        self.all_images.append(img_path)
                        self.all_masks.append(mask_path)
                        self.case_names.append(os.path.splitext(file)[0])
                    else:
                        # Strategy 3: Try 'masks' subfolder if images are in 'images'
                        # This is a simple heuristic; can be expanded
                        pass

        if len(self.all_images) == 0:
            print(f"Warning: No images found in {root_dir} with extension {img_ext} and mask suffix {mask_suffix}")
            
        # 2. Split the dataset
        # Use case_names to ensure consistent splitting
        indices = list(range(len(self.all_images)))
        
        # First split: Train vs (Val + Test)
        train_idx, temp_idx = train_test_split(
            indices, train_size=split_ratios[0], random_state=seed, shuffle=True
        )
        
        # Second split: Val vs Test
        # Normalize ratios for the remaining part
        remaining_ratio = split_ratios[1] + split_ratios[2]
        val_ratio_norm = split_ratios[1] / remaining_ratio
        
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=val_ratio_norm, random_state=seed, shuffle=True
        )
        
        # Select indices based on split
        if split == 'train':
            self.indices = train_idx
        elif split == 'val':
            self.indices = val_idx
        else:
            self.indices = test_idx
            
        print(f"[UniversalDataset] {split.upper()}: {len(self.indices)} samples from {root_dir}")
        
        # 3. Set Transforms
        if transform:
            self.transform = transform
        else:
            train_t, val_t = get_transformations(image_size)
            self.transform = train_t if split == 'train' else val_t

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        img_path = self.all_images[real_idx]
        mask_path = self.all_masks[real_idx]
        
        # Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            label = augmented['mask']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            label = torch.from_numpy(mask).long()
            
        # Ensure label is long
        if label.dtype != torch.long:
            label = label.long()
            
        return {
            'image': image,
            'label': label,
            'case_name': self.case_names[real_idx]
        }

class ConfigurableMedicalDataset(Dataset):
    """
    Dataset loader that reads from a YAML configuration file.
    Supports pre-split datasets (train/val/test in separate folders).
    """
    def __init__(self, config_path_or_dict, split='train', image_size=512, transform=None):
        """
        Args:
            config_path_or_dict: Path to .yaml file OR a dictionary containing config.
            split: 'train', 'val', or 'test'.
            image_size: Target size for resizing.
            transform: Optional custom transform. If None, uses default based on split.
        """
        self.split = split
        self.image_size = image_size
        
        # 1. Load Configuration
        if isinstance(config_path_or_dict, str):
            with open(config_path_or_dict, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config_path_or_dict

        root_dir = self.config.get('root_dir', '')
        
        # 2. Get paths for the specific split
        if split not in self.config:
            # Fallback: if 'test' is requested but not in config, use 'val' or raise error
            # For now, strict mode
            raise ValueError(f"Split '{split}' not found in configuration file.")
            
        split_config = self.config[split]
        
        # Handle relative or absolute paths
        img_dir = split_config['images']
        mask_dir = split_config['masks']
        
        if root_dir and not os.path.isabs(img_dir):
            img_dir = os.path.join(root_dir, img_dir)
        if root_dir and not os.path.isabs(mask_dir):
            mask_dir = os.path.join(root_dir, mask_dir)
            
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Image dir '{img_dir}' or Mask dir '{mask_dir}' does not exist.")

        # 3. Settings
        self.mask_suffix = self.config.get('mask_suffix', '')
        # Auto-detect common medical image formats (no need to define in YAML)
        valid_img_exts = set(['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.dcm', '.nii', '.nii.gz'])
        
        # 4. Match Images and Masks
        self.samples = []
        
        # Index all masks by stem (filename without extension)
        mask_map = {}
        for root, _, files in os.walk(mask_dir):
            for f in files:
                stem = os.path.splitext(f)[0]
                mask_map[stem] = os.path.join(root, f)
        
        # Iterate images and find matching masks
        for root, _, files in os.walk(img_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_img_exts:
                    img_stem = os.path.splitext(f)[0]
                    
                    # Construct expected mask stem
                    # e.g. Image: "case1", Suffix: "_seg" -> Mask: "case1_seg"
                    expected_mask_stem = img_stem + self.mask_suffix
                    
                    if expected_mask_stem in mask_map:
                        self.samples.append({
                            'image': os.path.join(root, f),
                            'label': mask_map[expected_mask_stem],
                            'case_name': img_stem
                        })
        
        if len(self.samples) == 0:
            raise ValueError(f"No matching pairs found for split '{split}'. Check paths and suffixes.")
            
        print(f"[ConfigurableDataset] Loaded {split.upper()}: {len(self.samples)} samples from {img_dir}")

        # 5. Set Transforms
        if transform:
            self.transforms = transform
        else:
            # Default transforms
            train_t, val_t = get_transformations(image_size)
            self.transforms = train_t if split == 'train' else val_t

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load Image
        image = cv2.imread(sample['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Mask
        mask = cv2.imread(sample['label'], cv2.IMREAD_GRAYSCALE)
        
        # Force binary mask (0/1) to avoid label out-of-range in CE loss
        mask = (mask > 0).astype(np.uint8)
        
        # Apply Transforms
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            label = augmented['mask']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            label = torch.from_numpy(mask).long()
            
        if label.dtype != torch.long:
            label = label.long()
            
        return {
            'image': image,
            'label': label,
            'case_name': sample['case_name']
        }

def get_dataset_from_config(config_path, split='train', image_size=512):
    """Helper to create dataset directly from yaml path"""
    return ConfigurableMedicalDataset(config_path, split=split, image_size=image_size)
