import openslide
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms



# 加载WSI和对应的标注（假设标注是二值化的PNG mask）
class WSIDataset(Dataset):
    def __init__(self, wsi_path, mask_path, patch_size=256, level=0):
        self.wsi = openslide.OpenSlide(wsi_path)
        print(mask_path)
        self.mask = openslide.OpenSlide(mask_path)
        self.level_count = self.mask.level_count
        # print(f"mask level count: {self.mask.level_count}")
        # raise RuntimeError("mask level count: {self.mask.level_count}")
        self.patch_size = patch_size
        self.level = level
        self.wsi_width, self.wsi_height = self.wsi.level_dimensions[level]
        self.mask_width, self.mask_height = self.mask.level_dimensions[level]

    def __len__(self):
        return (self.wsi_width // self.patch_size) * (self.wsi_height // self.patch_size)

    def __getitem__(self, idx):

        # 计算当前patch的坐标
        x = (idx % (self.wsi_width // self.patch_size)) * self.patch_size
        y = (idx // (self.wsi_width // self.patch_size)) * self.patch_size

        # 从WSI中提取patch
        patch = self.wsi.read_region((x, y), self.level, (self.patch_size, self.patch_size))
        patch = patch.convert('RGB')

        # 从mask中提取对应的区域
        mask_x = int(x * (self.mask_width / self.wsi_width))
        mask_y = int(y * (self.mask_height / self.wsi_height))

        print(f"Extracting patch at WSI coordinates ({x}, {y}) and mask coordinates ({mask_x}, {mask_y})")
        print(f"Patch size: {self.patch_size}, Mask size: {self.wsi_width}x{self.wsi_height}")
        mask_patch = self.mask.read_region((mask_x, mask_y), 0, (self.patch_size, self.patch_size))

        # 转为灰度图并二值化
        # [i,0,0] i=1,2,3,4,5,6六分类
        mask_patch = np.array(mask_patch)[..., :1]  # 只取0通道

            # print("inspect mask[:5] values:", mask_patch[:5])  # 检查mask的前5个值
            # if np.sum(mask_patch)>256*256:  # 检查mask的总和
            #     # print(np.unique(mask_patch))  # 检查mask中唯一值
            #     break
            # else:
            #     idx += 1  # 如果mask为空，继续下一个patch
        # thresh = threshold_otsu(mask_patch)
        # mask_patch = (mask_patch > thresh).astype(np.uint8) * 255

        # 转为numpy数组并归一化
        patch = np.array(patch) / 255.0
        mask_patch = np.array(mask_patch) / 255.0  # 假设mask是0-1二值图

        # 转为PyTorch张量
        patch = torch.from_numpy(patch).permute(2, 0, 1).float()
        mask_patch = torch.from_numpy(mask_patch).unsqueeze(0).float()

        return patch, mask_patch
    
    def collate_fn(self, batch):
        patches, masks = zip(*batch)
        patches = torch.stack(patches)
        masks = torch.stack(masks)
        return patches, masks
    
    def visualize(self, idx):
        # 可视化指定索引的patch和mask
        patch, mask = self.__getitem__(idx)
        patch = patch.permute(1, 2, 0).numpy()  #
        mask = mask.squeeze().numpy()
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        # plt.imshow(patch)
        plt.savefig("results/patch.png", dpi=300)
        plt.title('Patch')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        plt.axis('off')
        plt.savefig("results/mask.png", dpi=300)

# # 示例：加载WSI和mask
# if __name__ == "__main__":
#     wsi_path = "demo/706a5789a3517393a583829512a1fb8d.tiff"
#     mask_path = "demo/706a5789a3517393a583829512a1fb8d_mask.tiff"
#     level = 2  # 使用WSI的最高分辨率层级
#     dataset = WSIDataset(wsi_path, mask_path, patch_size=256, level=level)
#     print(f"Dataset size: {len(dataset)} patches")
#     for level_i in range(dataset.level_count):
#         print(f"Level {level_i} WSI dimensions: {dataset.wsi.level_dimensions[level_i]}")
#         print(f"Level {level_i} Mask dimensions: {dataset.mask.level_dimensions[level_i]}")
#     print(f"WSI dimensions: {dataset.wsi.level_dimensions[level]}")
#     print(f"Mask dimensions: {dataset.mask.level_dimensions[level]}")
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
#     for i, (patches, masks) in enumerate(dataloader):
#         if patches is None or masks is None:
#             continue
#         print(f"Batch {i+1}:")
#         print(f"Patches shape: {patches.shape}, Masks shape: {masks.shape}")
#         # if i == 0:  # 只可视化第一个batch
#         dataset.visualize(i)
#         if i>5:
#             break
#         # break  # 只处理一个batch以示例

# import os
# from torch.utils.data import random_split
# from sklearn.model_selection import train_test_split

# def create_datasets(data_dir, mask_dir, patch_size=256, level=2, val_ratio=0.1, test_ratio=0.1):
#     """
#     Create train/val/test datasets from directory of WSI files
    
#     Args:
#         data_dir: Path to directory containing WSI files (.tiff)
#         mask_dir: Path to directory containing corresponding mask files
#         patch_size: Size of patches to extract
#         level: Pyramid level to use
#         val_ratio: Proportion for validation set (default 10%)
#         test_ratio: Proportion for test set (default 10%)
    
#     Returns:
#         train_dataset, val_dataset, test_dataset
#     """
#     # Get list of WSI files
#     wsi_files = [f for f in os.listdir(data_dir) if f.endswith('.tiff')]
    
#     # Split into train+val and test first (holdout test set)
#     train_val_files, test_files = train_test_split(
#         wsi_files, test_size=test_ratio, random_state=42
#     )
    
#     # Then split train_val into train and validation
#     train_files, val_files = train_test_split(
#         train_val_files, test_size=val_ratio/(1-test_ratio), random_state=42
#     )
    
#     print(f"Total WSIs: {len(wsi_files)}")
#     print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
#     # Create datasets
#     train_dataset = WSIDataset.from_file_list(
#         train_files, data_dir, mask_dir, patch_size, level
#     )
#     val_dataset = WSIDataset.from_file_list(
#         val_files, data_dir, mask_dir, patch_size, level
#     )
#     test_dataset = WSIDataset.from_file_list(
#         test_files, data_dir, mask_dir, patch_size, level
#     )
    
#     return train_dataset, val_dataset, test_dataset

# if __name__ == "__main__":
#     # Configuration
#     data_dir = "data/wsi/train"
#     mask_dir = "data/wsi/labels"
#     patch_size = 256
#     level = 2
#     batch_size = 4
    
#     # Create datasets with 80% train, 10% val, 10% test split
#     train_dataset, val_dataset, test_dataset = create_datasets(
#         data_dir, mask_dir, patch_size, level, val_ratio=0.1, test_ratio=0.1
#     )
    
#     # Create dataloaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     # Example usage
#     print("\nTraining samples:")
#     for i, (patches, masks) in enumerate(train_loader):
#         print(f"Train Batch {i+1}: Patches {patches.shape}, Masks {masks.shape}")
#         if i > 2:  # Just show a few batches
#             break
    
#     print("\nValidation samples:")
#     for i, (patches, masks) in enumerate(val_loader):
#         print(f"Val Batch {i+1}: Patches {patches.shape}, Masks {masks.shape}")
#         if i > 2:
#             break


import os
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split

# def create_datasets(data_dir, mask_dir, patch_size=256, level=2, val_ratio=0.1, test_ratio=0.1):
#     """
#     Create train/val/test datasets from directory of WSI files
    
#     Args:
#         data_dir: Path to directory containing WSI files (.tiff)
#         mask_dir: Path to directory containing corresponding mask files
#         patch_size: Size of patches to extract
#         level: Pyramid level to use
#         val_ratio: Proportion for validation set (default 10%)
#         test_ratio: Proportion for test set (default 10%)
    
#     Returns:
#         train_dataset, val_dataset, test_dataset (lists of WSIDataset instances)
#     """
#     # Get list of WSI files
#     wsi_files = [f for f in os.listdir(data_dir) if f.endswith('.tiff')]
    
#     # Split into train+val and test first (holdout test set)
#     train_val_files, test_files = train_test_split(
#         wsi_files, test_size=test_ratio, random_state=42
#     )
    
#     # Then split train_val into train and validation
#     train_files, val_files = train_test_split(
#         train_val_files, test_size=val_ratio/(1-test_ratio), random_state=42
#     )
    
#     print(f"Total WSIs: {len(wsi_files)}")
#     print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
#     # Create datasets by initializing WSIDataset for each file
#     train_datasets = [
#         WSIDataset(
#             os.path.join(data_dir, f),
#             os.path.join(mask_dir, f.replace('.tiff', '_mask.tiff')),
#             patch_size=patch_size,
#             level=level
#         ) for f in train_files
#     ]
    
#     val_datasets = [
#         WSIDataset(
#             os.path.join(data_dir, f),
#             os.path.join(mask_dir, f.replace('.tiff', '_mask.tiff')),
#             patch_size=patch_size,
#             level=level
#         ) for f in val_files
#     ]
    
#     test_datasets = [
#         WSIDataset(
#             os.path.join(data_dir, f),
#             os.path.join(mask_dir, f.replace('.tiff', '_mask.tiff')),
#             patch_size=patch_size,
#             level=level
#         ) for f in test_files
#     ]
    
#     return train_datasets, val_datasets, test_datasets

def create_datasets(data_dir, mask_dir, patch_size=256, level=2, val_ratio=0.1, test_ratio=0.1):
    """
    Create train/val/test datasets from directory of WSI files, skipping WSIs without masks
    """
    # Get list of WSI files and verify corresponding masks exist
    valid_pairs = []
    for f in os.listdir(data_dir):
        if f.endswith('.tiff'):
            mask_path = os.path.join(mask_dir, f.replace('.tiff', '_mask.tiff'))
            if os.path.exists(mask_path):
                valid_pairs.append(f)
            else:
                print(f"Warning: Mask not found for {f}, skipping...")
    
    if not valid_pairs:
        raise ValueError("No valid WSI-mask pairs found in the directories")
    
    # Split into train+val and test first (holdout test set)
    train_val_files, test_files = train_test_split(
        valid_pairs, test_size=test_ratio, random_state=42
    )
    
    # Then split train_val into train and validation
    train_files, val_files = train_test_split(
        train_val_files, test_size=val_ratio/(1-test_ratio), random_state=42
    )
    
    print(f"\nDataset Summary:")
    print(f"Found {len(valid_pairs)} valid WSI-mask pairs")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    def create_dataset_safe(file_list):
        datasets = []
        for f in file_list:
            try:
                wsi_path = os.path.join(data_dir, f)
                mask_path = os.path.join(mask_dir, f.replace('.tiff', '_mask.tiff'))
                
                # Additional verification
                if not os.path.exists(wsi_path):
                    print(f"Warning: WSI file {wsi_path} not found, skipping")
                    continue
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask file {mask_path} not found, skipping")
                    continue
                
                dataset = WSIDataset(
                    wsi_path,
                    mask_path,
                    patch_size=patch_size,
                    level=level
                )
                # Skip empty datasets (WSIs with no valid patches)
                if len(dataset) > 0:
                    datasets.append(dataset)
                else:
                    print(f"Warning: No valid patches found in {f}, skipping")
            except Exception as e:
                print(f"Error loading {f}: {str(e)}, skipping")
                continue
        return datasets
    
    train_datasets = create_dataset_safe(train_files)
    val_datasets = create_dataset_safe(val_files)
    test_datasets = create_dataset_safe(test_files)
    
    if not train_datasets:
        raise ValueError("No valid training datasets created - check your data")
    
    return train_datasets, val_datasets, test_datasets

if __name__ == "__main__":
    # Configuration
    data_dir = "data/wsi/train"
    mask_dir = "data/wsi/labels"
    patch_size = 256
    level = 2
    batch_size = 4
    
    # Create datasets with 80% train, 10% val, 10% test split
    train_datasets, val_datasets, test_datasets = create_datasets(
        data_dir, mask_dir, patch_size, level, val_ratio=0.1, test_ratio=0.1
    )
    
    # Since we have multiple datasets, we can either:
    # 1. Use them separately (train on each sequentially)
    # 2. Combine them using torch.utils.data.ConcatDataset
    
    # Option 1: Process each WSI dataset separately
    print("\nProcessing train datasets:")
    for i, dataset in enumerate(train_datasets[:3]):  # Just show first 3 for demo
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print(f"\nTraining on WSI {i+1}/{len(train_datasets)}")
        for batch_idx, (patches, masks) in enumerate(loader):
            print(f"Batch {batch_idx+1}: Patches {patches.shape}, Masks {masks.shape}")
            if batch_idx > 2:  # Just show 3 batches per WSI
                break
    
    # Option 2: Combine all train datasets
    from torch.utils.data import ConcatDataset
    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets)
    combined_test = ConcatDataset(test_datasets)
    
    train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(combined_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(combined_test, batch_size=batch_size, shuffle=False)
    
    print("\nTraining on combined dataset:")
    for i, (patches, masks) in enumerate(train_loader):
        print(f"Batch {i+1}: Patches {patches.shape}, Masks {masks.shape}")
        if i > 2:  # Just show 3 batches
            break