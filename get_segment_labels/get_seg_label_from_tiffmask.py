import openslide
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb

def load_slide_and_mask(slide_path, mask_path, level=0):
    """加载WSI和对应的mask图像"""
    slide = openslide.OpenSlide(slide_path)
    mask = openslide.OpenSlide(mask_path)
    
    # 获取指定层级的尺寸
    slide_img = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB')
    mask_img = mask.read_region((0, 0), level, mask.level_dimensions[level]).convert('RGB')
    
    return np.array(slide_img), np.array(mask_img)

def create_highlight_mask(slide_img, mask_img, highlight_color=(255, 0, 0), alpha=0.3):
    """
    根据mask标注创建高亮覆盖层
    :param slide_img: 原始WSI图像(numpy数组)
    :param mask_img: mask图像(numpy数组)
    :param highlight_color: 高亮颜色(RGB)
    :param alpha: 透明度(0-1)
    :return: 带高亮标注的图像(PIL.Image)
    """
    # 提取mask的第0通道（假设标注在R通道）
    mask_data = mask_img[:, :, 0] > 0
    
    # 创建高亮层
    highlight = np.zeros_like(slide_img)
    highlight[mask_data] = highlight_color
    
    # 混合原图和高亮层
    blended = (slide_img * (1 - alpha) + highlight * alpha).astype(np.uint8)
    
    return Image.fromarray(blended)

def visualize_with_contours(slide_img, mask_img, color=(255, 0, 0), thickness=2):
    """
    用轮廓线方式标注（适用于稀疏标注）
    :param thickness: 轮廓线粗细(像素)
    """
    from skimage import measure
    
    # 提取mask并找到轮廓
    mask_data = mask_img[:, :, 0] > 0
    contours = measure.find_contours(mask_data, 0.5)
    
    # 在原图上绘制轮廓
    pil_img = Image.fromarray(slide_img)
    draw = ImageDraw.Draw(pil_img)
    
    for contour in contours:
        coords = [(int(x[1]), int(x[0])) for x in contour]  # (y,x) -> (x,y)
        draw.line(coords, fill=color, width=thickness)
    
    return pil_img

def adjust_hsv_highlight(slide_img, mask_img, hue_shift=0.7, sat_boost=1.5):
    """
    通过HSV色彩空间增强标注区域
    :param hue_shift: 色相偏移量(0-1)
    :param sat_boost: 饱和度增强倍数
    """
    # 转换到HSV空间
    hsv = rgb2hsv(slide_img.astype(float)/255)
    
    # 提取mask区域
    mask_data = mask_img[:, :, 0] > 0
    
    # 调整标注区域颜色
    hsv[mask_data, 0] = (hsv[mask_data, 0] + hue_shift) % 1.0  # 色相偏移
    hsv[mask_data, 1] = np.clip(hsv[mask_data, 1] * sat_boost, 0, 1)  # 增强饱和度
    
    # 转换回RGB
    highlighted = (hsv2rgb(hsv) * 255).astype(np.uint8)
    return Image.fromarray(highlighted)

# 示例使用
if __name__ == "__main__":
    # 文件路径（替换为实际路径）
    train_slide_path = "/data/PANDA_grading/train_images/10a745e5c034e37f75ebc8756fc2d6eb.tiff"
    mask_slide_path = "/data/PANDA_grading/train_label_masks/10a745e5c034e37f75ebc8756fc2d6eb_mask.tiff"
    
    # 加载图像（使用第3层级，尺寸较小适合快速处理）
    slide_img, mask_img = load_slide_and_mask(train_slide_path, mask_slide_path, level=1)
    
    # 方法1：半透明高亮覆盖
    highlighted = create_highlight_mask(slide_img, mask_img, 
                                     highlight_color=(255, 255, 0),  # 黄色高亮
                                     alpha=0.4)  # 40%透明度
    
    # 方法2：轮廓线标注
    contoured = visualize_with_contours(slide_img, mask_img, 
                                      color=(0, 255, 0),  # 绿色轮廓
                                      thickness=3)
    
    # 方法3：HSV色彩增强
    hsv_highlighted = adjust_hsv_highlight(slide_img, mask_img,
                                         hue_shift=0.3,  # 色相偏移
                                         sat_boost=2.0)  # 饱和度加倍
    
    # 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(highlighted)
    axes[0].set_title("透明高亮覆盖")
    axes[1].imshow(contoured)
    axes[1].set_title("轮廓线标注")
    axes[2].imshow(hsv_highlighted)
    axes[2].set_title("色彩增强")
    
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    # 保存结果
    highlighted.save("highlighted.png")
    contoured.save("contoured.png")
    hsv_highlighted.save("hsv_highlighted.png")