import os
import openslide
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import cv2

def visualize_annotated_patches(wsi_path, mask_path, output_dir, patch_size=512, level=2):
    """
    处理WSI和Mask WSI，生成带颜色标注的可视化Patch
    
    参数:
        wsi_path: WSI文件路径(.svs/.ndpi等)
        mask_path: 标注Mask文件路径
        output_dir: 输出目录
        patch_size: 裁剪尺寸(默认512)
        level: 金字塔层级(默认2)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开WSI和Mask
    wsi = openslide.OpenSlide(wsi_path)
    mask = openslide.OpenSlide(mask_path)
    
    # 获取指定层级的尺寸
    wsi_width, wsi_height = wsi.level_dimensions[level]
    mask_width, mask_height = mask.level_dimensions[level]
    
    # 计算缩放比例(确保WSI和Mask尺寸匹配)
    scale_x = mask_width / wsi_width
    scale_y = mask_height / wsi_height
    
    # 生成缩略图用于可视化
    thumbnail = wsi.get_thumbnail((wsi_width // 4, wsi_height // 4))
    thumbnail = np.array(thumbnail)
    
    # 定义标注颜色映射 (可根据实际标注修改)
    color_map = {
        0: [0, 255, 0],      # 背景: 绿色
        1: [255, 0, 0],    # 类别1: 红色
        2: [0, 0, 255],    # 类别2: 绿色
        3: [0, 0, 255],    # 类别3: 蓝色
        4: [255, 255, 0]   # 类别4: 黄色
    }
    
    # 遍历所有Patch
    for y in range(0, wsi_height, patch_size):
        for x in range(0, wsi_width, patch_size):
            # 计算实际裁剪尺寸(避免越界)
            curr_size = min(patch_size, wsi_width-x, wsi_height-y)
            
            # 读取WSI Patch
            wsi_patch = wsi.read_region((x, y), level, (curr_size, curr_size))
            wsi_patch = wsi_patch.convert('RGB')
            
            # 计算对应的Mask位置
            mask_x = int(x * scale_x)
            mask_y = int(y * scale_y)
            mask_patch = mask.read_region((mask_x, mask_y), level, (curr_size, curr_size))
            mask_patch = np.array(mask_patch.convert('L'))  # 转为灰度图
            
            # 创建彩色标注层
            colored_mask = np.zeros((curr_size, curr_size, 3), dtype=np.uint8)
            for label, color in color_map.items():
                colored_mask[mask_patch == label] = color
            
            # 将标注叠加到原图(透明度50%)
            overlay = cv2.addWeighted(np.array(wsi_patch), 0.7, colored_mask, 0.3, 0)
            
            # 保存结果
            patch_output = os.path.join(output_dir, f'patch_{x}_{y}.png')
            Image.fromarray(overlay).save(patch_output)
            
            # 在缩略图上标记位置
            thumb_x = int(x / wsi_width * thumbnail.shape[1])
            thumb_y = int(y / wsi_height * thumbnail.shape[0])
            thumb_size = max(2, int(patch_size / wsi_width * thumbnail.shape[1]))
            cv2.rectangle(thumbnail, (thumb_x, thumb_y), 
                         (thumb_x+thumb_size, thumb_y+thumb_size), 
                         (0, 255, 255), 2)
    
    # 保存带标记的缩略图
    overview_path = os.path.join(output_dir, 'overview.png')
    Image.fromarray(thumbnail).save(overview_path)
    
    # 显示示例结果
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(thumbnail)
    ax[0].set_title('WSI Overview with Patch Locations')
    
    sample_patch = Image.open(os.path.join(output_dir, os.listdir(output_dir)[0]))
    ax[1].imshow(sample_patch)
    ax[1].set_title('Sample Annotated Patch')
    
    plt.savefig(os.path.join(output_dir, 'summary.png'))
    plt.close()
    
    print(f"Processing complete. Results saved to {output_dir}")

# 使用示例
if __name__ == "__main__":
    visualize_annotated_patches(
        wsi_path="demo/706a5789a3517393a583829512a1fb8d.tiff",
        mask_path="demo/706a5789a3517393a583829512a1fb8d_mask.tiff",
        output_dir="results/visualization",
        patch_size=512,
        level=2
    )