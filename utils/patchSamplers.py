import cv2
import numpy as np
from abc import ABC, abstractmethod
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from skimage.morphology import remove_small_objects, opening, closing
from openslide import OpenSlide
from typing import List, Tuple
import matplotlib.pyplot as plt

class PatchSampler(ABC):
    """抽象基类定义采样器接口"""
    @abstractmethod
    def sample_patches(self, wsi: OpenSlide, num_patches: int) -> List[Tuple[int, int, int, int]]:
        pass

    @staticmethod
    def get_thumbnail(wsi: OpenSlide, size: int = 2048) -> np.ndarray:
        """获取WSI缩略图用于快速计算"""
        thumb = wsi.get_thumbnail((size, size))
        return np.array(thumb)
    
class RandomSampler(PatchSampler):
    """完全随机采样，作为基线方法"""
    def sample_patches(self, wsi: OpenSlide, num_patches: int, patch_size: int = 512) -> List[Tuple]:
        selected = []
        for _ in range(num_patches * 3):  # 超量采样保证数量
            x = np.random.randint(0, wsi.dimensions[0] - patch_size)
            y = np.random.randint(0, wsi.dimensions[1] - patch_size)
            selected.append((x, y, patch_size, patch_size))
        return selected[:num_patches]
    
class OtsuSampler(PatchSampler):
    """基于Otsu阈值排除空白区域"""
    def sample_patches(self, wsi: OpenSlide, num_patches: int, patch_size: int = 512) -> List[Tuple]:
        thumb = self.get_thumbnail(wsi)
        gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
        thresh = threshold_otsu(gray)
        mask = gray > thresh * 0.7  # 宽松阈值
        
        scale_x = wsi.dimensions[0] / thumb.shape[1]
        scale_y = wsi.dimensions[1] / thumb.shape[0]
        
        selected = []
        while len(selected) < num_patches:
            y, x = np.random.randint(0, thumb.shape[0]), np.random.randint(0, thumb.shape[1])
            if mask[y, x]:
                x_full = int(x * scale_x)
                y_full = int(y * scale_y)
                if x_full + patch_size < wsi.dimensions[0] and y_full + patch_size < wsi.dimensions[1]:
                    selected.append((x_full, y_full, patch_size, patch_size))
        return selected
    
class ClusteringSampler(PatchSampler):
    """K-means聚类识别不同组织类型"""
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters  # 通常3类：肿瘤/间质/正常

    def sample_patches(self, wsi: OpenSlide, num_patches: int, patch_size: int = 512) -> List[Tuple]:
        thumb = self.get_thumbnail(wsi)
        pixels = thumb.reshape((-1, 3)).astype(np.float32)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(pixels)
        cluster_map = labels.reshape(thumb.shape[:2])
        
        # 计算每类权重（假设cluster 0是肿瘤）
        weights = np.bincount(labels) / len(labels)
        tumor_cluster = np.argmin(kmeans.cluster_centers_.mean(axis=1))  # 最暗的簇
        
        scale_x = wsi.dimensions[0] / thumb.shape[1]
        scale_y = wsi.dimensions[1] / thumb.shape[0]
        
        selected = []
        while len(selected) < num_patches:
            # 按类别权重采样
            cluster = np.random.choice(self.n_clusters, p=weights)
            ys, xs = np.where(cluster_map == cluster)
            if len(xs) > 0:
                idx = np.random.randint(len(xs))
                x_full = int(xs[idx] * scale_x)
                y_full = int(ys[idx] * scale_y)
                if x_full + patch_size < wsi.dimensions[0] and y_full + patch_size < wsi.dimensions[1]:
                    selected.append((x_full, y_full, patch_size, patch_size))
        return selected
    
class MorphologySampler(PatchSampler):
    def sample_patches(self, wsi: OpenSlide, num_patches: int, patch_size: int = 512) -> List[Tuple]:
        thumb = self.get_thumbnail(wsi)
        gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
        
        # Otsu阈值 → 生成布尔mask
        thresh = threshold_otsu(gray)
        mask = gray > thresh  # 此时是布尔类型
        
        # 形态学操作（使用布尔类型）
        mask = remove_small_objects(mask, min_size=500)
        mask = opening(mask, np.ones((3,3)))
        mask = closing(mask, np.ones((7,7)))
        
        # 转换为uint8供OpenCV使用
        mask_uint8 = mask.astype(np.uint8) * 255  # 将True/False转换为0/255
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建ROI mask
        roi_mask = np.zeros_like(mask_uint8)
        cv2.drawContours(roi_mask, contours, -1, 255, thickness=cv2.FILLED)
        
        # 坐标转换和采样
        scale_x = wsi.dimensions[0] / thumb.shape[1]
        scale_y = wsi.dimensions[1] / thumb.shape[0]
        
        selected = []
        while len(selected) < num_patches:
            y, x = np.random.randint(0, thumb.shape[0]), np.random.randint(0, thumb.shape[1])
            if roi_mask[y, x] > 0:  # 检查是否为有效区域
                x_full = int(x * scale_x)
                y_full = int(y * scale_y)
                if x_full + patch_size < wsi.dimensions[0] and y_full + patch_size < wsi.dimensions[1]:
                    selected.append((x_full, y_full, patch_size, patch_size))
        
        return selected
    
def test_samplers(wsi_path: str, num_patches: int = 10):
    """对比测试所有采样器"""
    wsi = OpenSlide(wsi_path)
    samplers = {
        "Random": RandomSampler(),
        "Otsu": OtsuSampler(),
        "Clustering": ClusteringSampler(n_clusters=3),
        "Morphology": MorphologySampler()
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    for (name, sampler), ax in zip(samplers.items(), axes):
        patches = sampler.sample_patches(wsi, num_patches)
        thumb = sampler.get_thumbnail(wsi, 1024)
        
        # 在缩略图上绘制patch位置
        scale = thumb.shape[1] / wsi.dimensions[0]
        for x, y, w, h in patches:
            px, py = int(x * scale), int(y * scale)
            cv2.rectangle(thumb, (px, py), (px + int(w * scale), py + int(h * scale)), (255,0,0), 3)
        
        ax.imshow(thumb)
        ax.set_title(f"{name} Sampling")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("results/sampling_comparison.png", dpi=300)

class TissueClassifier:
    """基于聚类的组织分类器"""
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans = None
    
    def classify(self, patch: np.ndarray) -> int:
        """返回patch的类别标签"""
        if self.kmeans is None:
            raise ValueError("Model not trained!")
        features = patch.mean(axis=(0,1))  # 简单使用平均颜色
        return self.kmeans.predict([features])[0]
    
    def train(self, wsi: OpenSlide):
        """使用WSI训练分类器"""
        thumb = PatchSampler.get_thumbnail(wsi)
        pixels = thumb.reshape((-1, 3))
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        self.kmeans.fit(pixels)

def region_stats(patches: List[Tuple], classifier: TissueClassifier, wsi: OpenSlide):
    """统计各区域类型比例"""
    counts = {i:0 for i in range(classifier.n_clusters)}
    for x, y, w, h in patches:
        patch = np.array(wsi.read_region((x,y), 0, (w,h)))
        label = classifier.classify(patch)
        counts[label] += 1
    return counts

if __name__ == "__main__":
    wsi_path="demo/706a5789a3517393a583829512a1fb8d.tiff"
    wsi = OpenSlide(wsi_path)
    test_samplers(wsi_path, num_patches=15)
