import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import filters.infer as infer
from PIL import Image

# 定义类别颜色 (RGB格式)
class_colors = [
    [0, 0, 0],       # 背景 - 黑色
    [1, 0, 0],       # 类别1 - 红色
    [0, 1, 0],       # 类别2 - 绿色
    [0, 0, 1],       # 类别3 - 蓝色
    [1, 1, 0],       # 类别4 - 黄色
    [1, 0, 1],       # 类别5 - 品红
    [0, 1, 1]        # 类别6 - 青色
]

# 创建自定义colormap
custom_cmap = ListedColormap(class_colors)


def visualize_segmentation(image, gt_mask, pred_mask=None, class_names=None):
    """
    论文级分割可视化
    参数:
        image: [H,W,3] 原始图像
        gt_mask: [H,W] 或 [H,W,1] 标注mask (值1-6)
        pred_mask: [H,W] 模型预测mask (可选)
        class_names: 类别名称列表
    """
    if class_names is None:
        class_names = ['Background', 'Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6']
    
    image,gt_mask,pred_mask = map(lambda x: Image.open(x).convert('RGB') if isinstance(x, str) else x, 
                                  [image, gt_mask, pred_mask])
    # 确保mask为二维
    gt_mask = gt_mask.squeeze()
    if pred_mask is not None:
        pred_mask = pred_mask.squeeze()
    
    fig, axes = plt.subplots(1, 3 if pred_mask is not None else 2, 
                            figsize=(18, 6), dpi=100)
    axes = axes.flatten()
    
    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title("Input Image", fontsize=12, pad=10)
    axes[0].axis('off')
    
    # 标注mask
    im = axes[1].imshow(gt_mask, cmap=custom_cmap, vmin=0, vmax=6)
    axes[1].set_title("Ground Truth", fontsize=12, pad=10)
    axes[1].axis('off')
    
    # 预测mask (如果有)
    if pred_mask is not None:
        im = axes[2].imshow(pred_mask, cmap=custom_cmap, vmin=0, vmax=6)
        axes[2].set_title("Prediction", fontsize=12, pad=10)
        axes[2].axis('off')
    
    # 添加colorbar
    cbar = fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    cbar.set_ticks(np.arange(6) + 0.5)
    cbar.set_ticklabels(class_names)
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.show()

# 模拟数据
# image = np.random.rand(256, 256, 3)  # 假图像 [H,W,3]
# gt_mask = np.random.randint(1, 7, (256, 256, 1))  # 假标注 [H,W,1]
# pred_mask = np.random.randint(0, 7, (256, 256))   # 假预测 [H,W]

model = infer.get_model()
# 使用示例
from filters.train import create_loader  # 替换为你的数据集加载函数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, val_loader, test_loader = create_loader(
    data_dir="data/wsi/train",
    mask_dir="data/wsi/labels",
    patch_size=256,
    level=2,
    batch_size=1,
    shuffle=True
)
with torch.no_grad():
    gt_image, gt_mask = next(iter(val_loader))
    print(f"Processing batch with images shape: {gt_image.shape}, masks shape: {gt_mask.shape}")
    pred_mask = model(gt_image.to(device))[0]

gt_image = gt_image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [H,W,3]
gt_mask = gt_mask.squeeze(0).squeeze(1).cpu().numpy()  # [H,W]
pred_mask = pred_mask.squeeze(0).squeeze(1).cpu().numpy()  # [H,W]
# 自定义类别名称
class_names = [
    'Back', 'Tumor', 'Stroma', 
    'Lymphocyte', 'Necrosis', 'Vessel', 'Other'
]

# 可视化
visualize_segmentation(
    image=gt_image,
    gt_mask=gt_mask,
    pred_mask=pred_mask,
    class_names=class_names
)

from skimage.segmentation import mark_boundaries

def visualize_with_boundaries(image, gt_mask, pred_mask):
    """带边界叠加的可视化"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 处理mask
    gt_mask = gt_mask.squeeze()
    pred_mask = pred_mask.squeeze()
    
    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    
    # 标注mask边界
    gt_boundary = mark_boundaries(
        image, 
        gt_mask, 
        color=(1,0,0),  # 红色边界
        mode='thick'
    )
    axes[1].imshow(gt_boundary)
    axes[1].set_title("GT Boundaries")
    
    # 预测mask边界
    pred_boundary = mark_boundaries(
        image, 
        pred_mask, 
        color=(0,1,0),  # 绿色边界
        mode='thick'
    )
    axes[2].imshow(pred_boundary)
    axes[2].set_title("Pred Boundaries")
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_class_iou(iou_scores, class_names):
    """绘制各类别IoU柱状图"""
    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(iou_scores)), iou_scores, color=class_colors[1:])
    
    plt.title("Per-class IoU Scores", fontsize=14)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("IoU", fontsize=12)
    plt.xticks(range(len(iou_scores)), class_names[1:], rotation=45)
    plt.ylim(0, 1)
    
    # 在柱子上方显示数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# 使用示例
iou_scores = [0.85, 0.72, 0.68, 0.91, 0.55, 0.78]
plot_class_iou(iou_scores, class_names)


fig = plt.figure(figsize=(15, 10), constrained_layout=True)
gs = fig.add_gridspec(2, 3)

# 第一行：原始图像 + GT + 预测
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

# 第二行：边界叠加 + IoU统计
ax4 = fig.add_subplot(gs[1, :2])
ax5 = fig.add_subplot(gs[1, 2])

# ... 在各ax上绘制内容 ...
plt.savefig('paper_ready.png', dpi=300, bbox_inches='tight')