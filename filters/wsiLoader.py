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
        self.mask = openslide.OpenSlide(mask_path)
        self.patch_size = patch_size
        self.level = level
        self.wsi_width, self.wsi_height = self.wsi.level_dimensions[level]
        self.mask_width, self.mask_height = self.mask.level_dimensions[level]

    def __len__(self):
        return (self.wsi_width // self.patch_size) * (self.wsi_height // self.patch_size)

    def __getitem__(self, idx):
        while True:
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
            if np.sum(mask_patch)>256*256:  # 检查mask的总和
                # print(np.unique(mask_patch))  # 检查mask中唯一值
                break
            else:
                idx += 1  # 如果mask为空，继续下一个patch
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

# 示例：加载WSI和mask
if __name__ == "__main__":
    wsi_path = "demo/706a5789a3517393a583829512a1fb8d.tiff"
    mask_path = "demo/706a5789a3517393a583829512a1fb8d_mask.tiff"
    level = 0  # 使用WSI的最高分辨率层级
    dataset = WSIDataset(wsi_path, mask_path, patch_size=256, level=level)
    print(f"Dataset size: {len(dataset)} patches")
    print(f"WSI dimensions: {dataset.wsi.level_dimensions[level]}")
    print(f"Mask dimensions: {dataset.mask.level_dimensions[level]}")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for i, (patches, masks) in enumerate(dataloader):
        if patches is None or masks is None:
            continue
        print(f"Batch {i+1}:")
        print(f"Patches shape: {patches.shape}, Masks shape: {masks.shape}")
        # if i == 0:  # 只可视化第一个batch
        dataset.visualize(i)
        # break  # 只处理一个batch以示例