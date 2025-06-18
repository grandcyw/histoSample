import openslide
import numpy as np
from PIL import Image, ImageDraw
import cv2
from skimage.measure import shannon_entropy
from scipy.ndimage import uniform_filter
import openslide
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from skimage.color import rgb2hsv, hsv2rgb
import sys

class WSIAnnotator:
    def __init__(self, slide_path, mask_path):
        """初始化WSI和mask读取器"""
        self.slide = openslide.OpenSlide(slide_path)
        self.mask = openslide.OpenSlide(mask_path)
        assert self.slide.level_dimensions[0] == self.mask.level_dimensions[0], "Slide和Mask尺寸不匹配"
        
    def find_most_variable_region(self, window_size=2048, level=2, channel=0):
        """
        找到mask第0通道变化最剧烈的区域
        :param window_size: 目标区域大小(level=0下)
        :param level: 用于搜索的金字塔层级
        :param channel: 分析的mask通道
        :return: (x, y) level=0下的坐标
        """
        # 读取mask的指定层级
        scale = self.mask.level_downsamples[level]
        mask_level = self.mask.read_region((0, 0), level, self.mask.level_dimensions[level])
        mask_arr = np.array(mask_level)[:, :, channel]
        
        # 计算局部熵（变化程度度量）
        entropy_map = uniform_filter(mask_arr, size=int(window_size/scale))
        entropy_map = np.array([shannon_entropy(mask_arr[i:i+int(window_size/scale), j:j+int(window_size/scale)]) 
                              for i in range(0, mask_arr.shape[0]-int(window_size/scale), int(window_size/scale//4))
                              for j in range(0, mask_arr.shape[1]-int(window_size/scale), int(window_size/scale//4))])
        
        # 找到熵最大的窗口
        max_idx = np.argmax(entropy_map)
        max_row = (max_idx // (mask_arr.shape[1]//int(window_size/scale//4))) * int(window_size/scale//4)
        max_col = (max_idx % (mask_arr.shape[1]//int(window_size/scale//4))) * int(window_size/scale//4)
        
        # 转换到level=0坐标
        x = int(max_col * scale)
        y = int(max_row * scale)
        
        # 确保不超出边界
        x = min(x, self.mask.level_dimensions[0][0] - window_size)
        y = min(y, self.mask.level_dimensions[0][1] - window_size)
        
        return x, y
    
    def annotate_region(self, location, size=(2048, 2048), level=0, 
                      color_map={0:(0,0,0,0), 1:(255,0,0,100), 2:(0,255,0,100), 
                               3:(0,0,255,100), 4:(255,255,0,100), 5:(255,0,255,100)}):
        """
        标注指定区域
        :param location: (x, y) level=0坐标
        :param size: (w, h) 区域大小
        :param level: 读取层级
        :param color_map: 各mask值对应的颜色(R,G,B,A)
        :return: 标注后的PIL.Image
        """
        # 读取区域
        slide_region = self.slide.read_region(location, level, size).convert('RGBA')
        mask_region = self.mask.read_region(location, level, size).convert('RGB')
        mask_arr = np.array(mask_region)[:, :, 0]  # 使用第0通道
        
        # 创建标注层
        annotation = Image.new('RGBA', size, (0,0,0,0))
        draw = ImageDraw.Draw(annotation)
        
        # 为每个mask值创建标注
        for value, color in color_map.items():
            if value == 0:  # 忽略背景
                continue
            
            # 找到当前值的区域
            mask = (mask_arr == value)
            if not np.any(mask):
                continue
                
            # 绘制轮廓（更清晰的可视化）
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                coords = [(int(pt[0][0]), int(pt[0][1])) for pt in contour]
                if len(coords) > 2:  # 需要至少3个点构成多边形
                    draw.polygon(coords, fill=color, outline=(color[0],color[1],color[2],200))
        
        # 合并标注和原图
        result = Image.alpha_composite(slide_region, annotation)
        return result

    def process(self):
        """完整处理流程"""
        # 1. 找到变化最剧烈的区域
        x, y = self.find_most_variable_region()
        print(f"Selected ROI at level=0 coordinates: ({x}, {y})")
        
        # 2. 生成标注图像
        result = self.annotate_region((x, y))
        
        # 3. 添加图例
        legend = self.create_legend()
        final_img = Image.new('RGBA', (result.width, result.height + 50))
        final_img.paste(result, (0, 0))
        final_img.paste(legend, (10, result.height + 5))
        
        return final_img
    
    def create_legend(self):
        """创建图例"""
        legend = Image.new('RGBA', (800, 40), (255,255,255,200))
        draw = ImageDraw.Draw(legend)
        
        colors = {
            1: "红色: 类别1",
            2: "绿色: 类别2", 
            3: "蓝色: 类别3",
            4: "黄色: 类别4",
            5: "紫色: 类别5"
        }
        
        x_offset = 10
        for value, text in colors.items():
            color = (255,0,0,100) if value == 1 else \
                   (0,255,0,100) if value == 2 else \
                   (0,0,255,100) if value == 3 else \
                   (255,255,0,100) if value == 4 else \
                   (255,0,255,100)
            draw.rectangle([x_offset, 10, x_offset+20, 30], fill=color)
            draw.text((x_offset+25, 10), text, fill='black')
            x_offset += 150
        
        return legend

# 使用示例
if __name__ == "__main__":

    sample="6aff87e11871f4ce9682eec497239c71"
    # 初始化
    annotator = WSIAnnotator(slide_path = f"/data/PANDA_grading/train_images/{sample}.tiff",mask_path = f"/data/PANDA_grading/train_label_masks/{sample}_mask.tiff")

    x=17536
    y=15536
    # 执行处理
    result = annotator.process()
    # # 自定义颜色映射（RGBA格式）
    # custom_colors = {
    #     0: (0,0,0,0),      # 透明背景
    #     1: (255,100,100,80), # 浅红色
    #     2: (100,255,100,80), # 浅绿色
    #     3: (100,100,255,80), # 浅蓝色
    #     4: (255,200,50,80),  # 橙黄色
    #     5: (200,50,255,80)   # 粉紫色
    # }

    # # 使用自定义颜色
    # result = annotator.annotate_region((x,y), color_map=custom_colors)    
    # 显示结果
    plt.figure(figsize=(12, 12))
    plt.imshow(result)
    plt.axis('off')
    plt.title("最显著区域的多类别标注")
    plt.show()
    
    # 保存结果
    result.save("annotated_region.png")