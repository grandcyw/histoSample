import cv2
import numpy as np
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from skimage.morphology import remove_small_objects
import openslide

def wsi_smart_patching(wsi, patch_size=512, num_patches=10):
    """
    智能采样WSI中的有效patch
    
    参数:
        wsi: OpenSlide对象或WSI的缩略图数组
        patch_size: patch边长 (默认512)
        num_patches: 目标patch数量 (默认10)
    
    返回:
        selected_patches: [(x, y, w, h), ...]
    """
    # 生成低分辨率缩略图用于计算
    thumb = wsi.get_thumbnail((2048, 2048))
    thumb_gray = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2GRAY)
    
    # 方法1: Otsu阈值法排除空白区域
    otsu_thresh = threshold_otsu(thumb_gray)
    mask = thumb_gray > otsu_thresh * 0.8  # 宽松阈值
    
    # 方法2: 形态学优化
    mask = remove_small_objects(mask, min_size=500)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5,5)))
    
    # 方法3: K-means聚类识别高信息量区域
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            roi_centers.append((int(M['m10']/M['m00']), int(M['m01']/M['m00'])))
    
    # 方法4: 蒙特卡洛采样与验证
    selected = []
    scale_x = wsi.dimensions[0] / thumb.size[0]
    scale_y = wsi.dimensions[1] / thumb.size[1]
    
    while len(selected) < num_patches:
        # 优先在聚类中心附近采样
        if roi_centers and np.random.rand() > 0.3:
            center = roi_centers[np.random.randint(len(roi_centers))]
            x = np.random.normal(center[0], 100)
            y = np.random.normal(center[1], 100)
        else:
            x, y = np.random.randint(0, thumb.size[0]), np.random.randint(0, thumb.size[1])
        
        # 转换到全分辨率坐标
        x_full = int(x * scale_x)
        y_full = int(y * scale_y)
        
        # 验证patch有效性
        if mask[y,x] > 0 and \
           x_full + patch_size < wsi.dimensions[0] and \
           y_full + patch_size < wsi.dimensions[1]:
            selected.append((x_full, y_full, patch_size, patch_size))
    
    return selected[:num_patches]

if __name__ == "__main__":
    wsi_path = "demo/706a5789a3517393a583829512a1fb8d.tiff"
    slide = openslide.OpenSlide(wsi_path)
    
    patches = wsi_smart_patching(slide, patch_size=512, num_patches=10)
    
    print("Selected patches:")
    for patch in patches:
        print(patch)
    
    slide.close()