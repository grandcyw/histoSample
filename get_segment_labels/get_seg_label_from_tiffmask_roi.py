import openslide
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from skimage.color import rgb2hsv, hsv2rgb
import sys

import yacs

class WSIAnalyzer:
    def __init__(self, slide_path, mask_path):
        """初始化加载WSI和mask"""
        self.slide = openslide.OpenSlide(slide_path)
        self.mask = openslide.OpenSlide(mask_path)
        assert self.slide.level_count == self.mask.level_count, "Slide和Mask层级不匹配"
        
    def get_region_with_highlights(self, location, size, level=0, 
                                 highlight_params=None):
        """
        获取指定区域并添加高亮标注
        :param location: (x, y) 左上角坐标
        :param size: (w, h) 区域大小
        :param level: 金字塔层级
        :param highlight_params: 标注参数配置
        :return: PIL.Image对象
        """
        # 默认标注参数
        default_params = {
            'mask_channel': 0,       # 使用mask的R通道
            'highlight_color': (255, 0, 0),  # 红色半透明覆盖
            'contour_color': (0, 255, 0),    # 绿色轮廓线
            'hsv_shift': (0.7, 1.5), # (hue_shift, sat_boost)
            'alpha': 0.3,            # 透明度
            'contour_thickness': 2   # 轮廓线粗细
        }
        params = {**default_params, **(highlight_params or {})}
        
        # 读取指定区域
        slide_region = self.slide.read_region(location, level, size).convert('RGB')
        mask_region = self.mask.read_region(location, level, size).convert('RGB')
        slide_arr = np.array(slide_region)
        mask_arr = np.array(mask_region)
        
        # 提取标注区域（多通道mask支持）
        mask_data = mask_arr[:, :, params['mask_channel']]>0
        print("mask_data.shape",mask_data.shape)
        print(mask_data)
        # 创建空白画布
        result = Image.fromarray(slide_arr.copy())
        
        # 1. 半透明高亮覆盖
        if params.get('enable_highlight', True):
            highlight = Image.new('RGBA', size, (*params['highlight_color'], int(255*params['alpha'])))
            mask_layer = Image.fromarray((mask_data * 255).astype(np.uint8))
            result.paste(highlight, (0, 0), mask_layer)
        
        # 2. 轮廓线标注
        if params.get('enable_contour', True):
            contours = find_contours(mask_data, 0.5)
            draw = ImageDraw.Draw(result)
            for contour in contours:
                coords = [(int(p[1]), int(p[0])) for p in contour]  # (row,col)->(x,y)
                draw.line(coords, fill=params['contour_color'], width=params['contour_thickness'])
        
        # 3. HSV色彩增强
        if params.get('enable_hsv', False):
            hsv = rgb2hsv(slide_arr.astype(float)/255)
            hsv[mask_data, 0] = (hsv[mask_data, 0] + params['hsv_shift'][0]) % 1.0
            hsv[mask_data, 1] = np.clip(hsv[mask_data, 1] * params['hsv_shift'][1], 0, 1)
            enhanced = (hsv2rgb(hsv) * 255).astype(np.uint8)
            result = Image.blend(result, Image.fromarray(enhanced), 0.5)
        
        return result

# 使用示例
if __name__ == "__main__":
    # 1. 初始化分析器
    analyzer = WSIAnalyzer(slide_path = "/data/PANDA_grading/train_images/c4f94dca8de4d7d1936cc955512fe4bc.tiff",mask_path = "/data/PANDA_grading/train_label_masks/c4f94dca8de4d7d1936cc955512fe4bc_mask.tiff")
    x=17536-1000
    y=15536-1000
    mylevel=0
    # 2. 定义感兴趣区域（ROI）
    roi_location = (x, y)  # 左上角坐标(单位：level=0的像素)
    roi_size = (768*3, 768*3)      # 区域大小
    
    slide=openslide.OpenSlide("/data/PANDA_grading/train_images/c4f94dca8de4d7d1936cc955512fe4bc.tiff")
    slide.read_region((x, y), mylevel, (768*3, 768*3)).save("debug_region.png")
    # sys.exit(0)
    # 3. 获取带多种标注的区域图像
    highlighted = analyzer.get_region_with_highlights(
        location=roi_location,
        size=roi_size,
        level=mylevel,
        highlight_params={
            'highlight_color': (255, 255, 0),  # 黄色高亮
            'contour_color': (255, 0, 255),    # 紫色轮廓
            'alpha': 0.4,
            'enable_hsv': True
        }
    )
    
    # 4. 显示和保存结果
    plt.figure(figsize=(10, 10))
    plt.imshow(highlighted)
    plt.axis('off')
    plt.title("高分辨率区域标注示例")
    plt.show()
    highlighted.save("highlighted_roi.png")
    # 不同标注风格的参数配置示例
    styles = {
        # 病理学家偏好风格
        'pathology': {
            'highlight_color': (255, 0, 0),
            'contour_color': (0, 255, 0),
            'alpha': 0.2,
            'contour_thickness': 3
        },
        # 放射学偏好风格
        'radiology': {
            'highlight_color': (0, 255, 255),
            'enable_highlight': False,
            'contour_thickness': 4
        },
        # 量化分析风格
        'quantitative': {
            'enable_contour': False,
            'hsv_shift': (0.5, 2.0),
            'alpha': 0.15
        }
    }

    # 应用不同风格
    for style_name, params in styles.items():
        img = analyzer.get_region_with_highlights(roi_location, roi_size, level=mylevel, highlight_params=params)
        img.save(f"highlighted_{style_name}.png")