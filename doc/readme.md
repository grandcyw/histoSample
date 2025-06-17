以下是一个基于 **DenseNet** 对 **WSI（Whole Slide Image）** 的 **patch** 进行特征提取，并整合为整张 WSI 分类结果的完整项目架构和代码示例。我们将使用 **PyTorch** 实现，并包含以下关键步骤：

---

## **1. 项目架构**
### **整体流程**
1. **WSI 预处理**：使用 OpenSlide 或 libvips 将 WSI 切割成 patch（如 256x256）。
2. **Patch 特征提取**：用预训练的 DenseNet 提取每个 patch 的特征（如 1024 维向量）。
3. **特征聚合**：使用 **注意力机制（Attention）** 或 **平均池化（Mean Pooling）** 整合所有 patch 特征。
4. **WSI 分类**：训练一个分类器（如 MLP）预测 WSI 的最终标签。

### **目录结构**
```
wsi_classification/
├── data/
│   ├── wsi/                # 存放原始 WSI (.svs, .tif)
│   └── patches/            # 存放切割后的 patch (.png)
├── models/
│   ├── densenet.py         # DenseNet 特征提取模型
│   └── attention.py        # 注意力聚合模块
├── utils/
│   ├── patch_extractor.py  # WSI 切割工具
│   └── dataset.py          # 数据加载器
├── train.py                # 训练脚本
├── inference.py            # 推理脚本
└── config.yaml             # 配置文件
```

---

## **2. 核心代码实现**
### **(1) Patch 提取（`utils/patch_extractor.py`）**
使用 OpenSlide 切割 WSI：
```python
import openslide
import os
from PIL import Image

def extract_patches(wsi_path, output_dir, patch_size=256):
    slide = openslide.OpenSlide(wsi_path)
    width, height = slide.dimensions
    os.makedirs(output_dir, exist_ok=True)
    
    for x in range(0, width, patch_size):
        for y in range(0, height, patch_size):
            patch = slide.read_region((x, y), 0, (patch_size, patch_size))
            patch = patch.convert("RGB")
            patch.save(f"{output_dir}/patch_{x}_{y}.png")
```

---

### **(2) DenseNet 特征提取（`models/densenet.py`）**
加载预训练的 DenseNet-121 并提取特征：
```python
import torch
import torch.nn as nn
from torchvision.models import densenet121

class DenseNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = densenet121(pretrained=True)
        self.features = nn.Sequential(
            *list(self.model.children())[:-1]  # 移除最后的分类层
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.features(x)
        features = self.avgpool(features)
        return features.squeeze()  # 输出形状: [batch_size, 1024]
```

---

### **(3) 注意力聚合（`models/attention.py`）**
```python
import torch
import torch.nn as nn

class AttentionAggregator(nn.Module):
    def __init__(self, feature_dim=1024, num_classes=2):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=0)  # 对 patch 加权
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        # features: [num_patches, 1024]
        weights = self.attention(features)  # [num_patches, 1]
        weighted_features = torch.sum(weights * features, dim=0)  # [1024]
        logits = self.classifier(weighted_features)  # [num_classes]
        return logits
```

---

### **(4) 数据加载器（`utils/dataset.py`）**
```python
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class WSIDataset(Dataset):
    def __init__(self, patch_dir, transform=None):
        self.patch_paths = [os.path.join(patch_dir, f) for f in os.listdir(patch_dir)]
        self.transform = transform or T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        img = Image.open(self.patch_paths[idx])
        return self.transform(img)
```

---

### **(5) 训练脚本（`train.py`）**
```python
import torch
from torch.utils.data import DataLoader
from models.densenet import DenseNetFeatureExtractor
from models.attention import AttentionAggregator
from utils.dataset import WSIDataset

# 初始化模型
feature_extractor = DenseNetFeatureExtractor().cuda()
aggregator = AttentionAggregator().cuda()

# 数据加载
dataset = WSIDataset("data/patches/")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练循环
optimizer = torch.optim.Adam(aggregator.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch in dataloader:
        batch = batch.cuda()
        features = feature_extractor(batch)  # [batch_size, 1024]
        logits = aggregator(features)       # [batch_size, num_classes]
        
        # 假设标签为 0 或 1（需根据实际数据调整）
        labels = torch.zeros(len(batch)).long().cuda()  
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

### **(6) 推理脚本（`inference.py`）**
```python
def predict_wsi(wsi_path):
    # 1. 切割 WSI 为 patch
    extract_patches(wsi_path, "temp_patches/")
    
    # 2. 提取所有 patch 特征
    dataset = WSIDataset("temp_patches/")
    dataloader = DataLoader(dataset, batch_size=32)
    
    all_features = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.cuda()
            features = feature_extractor(batch)
            all_features.append(features)
    all_features = torch.cat(all_features, dim=0)  # [num_patches, 1024]
    
    # 3. 聚合特征并预测
    logits = aggregator(all_features)
    prob = torch.softmax(logits, dim=0)
    return prob.argmax().item()  # 返回预测类别
```

---

## **3. 关键优化点**
1. **多尺度 Patch**：提取不同放大级别的 patch（如 20x 和 40x）以捕获更多信息。
2. **GPU 加速**：使用混合精度训练（`torch.cuda.amp`）加速特征提取。
3. **负样本过滤**：跳过空白或无组织区域（通过 Otsu 阈值法）。
4. **分布式训练**：多 GPU 并行处理 patch。

---

## **4. 扩展方向**
- **自监督预训练**：用 SimCLR 或 MoCo 预训练 DenseNet。
- **图神经网络（GNN）**：将 patch 视为图节点，用 GNN 建模空间关系。
- **可解释性**：可视化注意力权重，定位关键区域（如肿瘤区域）。

---

通过以上架构，你可以高效地实现 **WSI → Patch → 特征 → 分类** 的完整流程。根据实际数据调整超参数（如 patch 大小、学习率）即可应用于具体任务（如癌症分级）。