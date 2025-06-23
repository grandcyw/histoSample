import argparse
import torch
import torch.nn as nn
from torchvision import models
from train import create_loader
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from models.unet2 import UNet  # 假设有独立的UNet实现

def parse_args():
    parser = argparse.ArgumentParser(description='Gleason Grading Segmentation Training')
    
    # 模型选择
    parser.add_argument('--model', type=str, default='simplecnn',
                       choices=['unet', 'simplecnn', 'densenet121'],
                       help='Model architecture (default: unet)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Input batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    
    # 数据参数
    parser.add_argument('--image-dir', type=str, required=False,
                       help='Path to training images directory')
    parser.add_argument('--mask-dir', type=str, required=False,
                       help='Path to corresponding masks directory')
    parser.add_argument('--num-classes', type=int, default=6,
                       help='Number of Gleason grades (default: 6)')
    
    # 实验管理
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Directory to save outputs (default: ./output)')
    parser.add_argument('--save-best', action='store_true',
                       help='Save best model based on validation loss')
    
    args = parser.parse_args()
    return args

# 模型定义
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(128, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def get_model(args):
    print(f"Using model: {args.model}")
    if args.model == 'unet':
        from models.unet import UNet  # 假设有独立的UNet实现
        return UNet(in_channels=3, out_channels=args.num_classes)
    elif args.model == 'simplecnn':
        return SimpleCNN(in_channels=3, num_classes=args.num_classes)
    elif args.model == 'densenet121':
        model = models.densenet121(pretrained=True)
        # 修改DenseNet最后一层用于分割
        model.features[-1] = nn.Sequential(
            nn.Conv2d(1024, args.num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        return model
    else:
        raise ValueError(f"Unknown model: {args.model}")

class SegmentationLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha  # Dice和BCE的权重
        
    def forward(self, pred, target):
        # 输入形状：
        # pred: [B,1,H,W] (经过sigmoid)
        # target: [B,1,H,W] (值为0或1)
        
        # Binary Cross Entropy
        bce_loss = F.binary_cross_entropy_with_logits(pred, target.float())
        
        # Dice Loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        union = pred_sigmoid.sum() + target.sum()
        dice_loss = 1 - (2. * intersection + 1) / (union + 1)  # 平滑处理
        
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

def iou_score(pred, target):
    # 计算IoU (Jaccard指数)
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)  # 避免除零


# def main():
#     args = parse_args()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # 1. 准备数据
#     # dataset = SegmentationDataset(args.image_dir, args.mask_dir)
#     # train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
#     # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
#     # val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

#     train_loader, val_loader, test_loader = create_loader(data_dir="data/wsi/train", mask_dir="data/wsi/labels", patch_size=256, level=2, batch_size=4, shuffle=True)

    
#     # 2. 初始化模型
#     model = get_model(args).to(device)
#     # criterion = nn.BCELoss()
#     criterion = SegmentationLoss(alpha=0.5)  # 使用自定义损失函数
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
#     # 3. 训练循环
#     best_loss = float('inf')
#     for epoch in range(args.epochs):
#         model.train()
#         train_loss = 0.0
        
#         for images, masks in train_loader:
#             print(f"Processing batch with images shape: {images.shape}, masks shape: {masks.shape}")
#             # images = images.permute(0, 3, 1, 2).float().to(device)
#             images = images.float().to(device)
#             print(f"Permuted images shape: {images.shape}")
#             masks = masks.squeeze().float().to(device)
            
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, masks)
#             loss.backward()
#             optimizer.step()
            
#             train_loss += loss.item()
        
#         # 验证
#         val_loss = validate(model, val_loader, criterion, device)
        
#         print(f'Epoch {epoch+1}/{args.epochs} | '
#               f'Train Loss: {train_loss/len(train_loader):.4f} | '
#               f'Val Loss: {val_loss:.4f}')
        
#         # 保存最佳模型
#         if args.save_best and val_loss < best_loss:
#             best_loss = val_loss
#             torch.save(model.state_dict(), f'{args.output_dir}/best_model.pth')
#             print('Saved best model!')

# def validate(model, val_loader, criterion, device):
#     model.eval()
#     val_loss = 0.0
    
#     with torch.no_grad():
#         for images, masks in val_loader:
#             images = images.permute(0, 3, 1, 2).float().to(device)
#             masks = masks.unsqueeze(1).float().to(device)
            
#             outputs = model(images)
#             val_loss += criterion(outputs, masks).item()
    
#     model.train()
#     return val_loss / len(val_loader)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    
    for images, masks in tqdm(loader, desc="Training"):
        images = images.to(device)          # [B,3,256,256]
        masks = masks.squeeze(-1).to(device)            # [B,1,256,256]
        
        # 前向传播
        outputs = model(images)             # [B,1,256,256]
        loss = criterion(outputs, masks)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算指标
        total_loss += loss.item()
        total_iou += iou_score(outputs, masks).item()
    
    return total_loss / len(loader), total_iou / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.to(device)
            masks = masks.squeeze(-1).to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            total_iou += iou_score(outputs, masks).item()
    
    return total_loss / len(loader), total_iou / len(loader)

def main():
    best_iou = 0.0
    args = parse_args()
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = SegmentationLoss(alpha=0.7)  # 70% BCE + 30% Dice
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 数据加载
    train_loader, val_loader, test_loader = create_loader(data_dir="data/wsi/train", mask_dir="data/wsi/labels", patch_size=256, level=2, batch_size=4, shuffle=True)
    
    print(f"Train loader size: {len(train_loader)}, Val loader size: {len(val_loader)}, Test loader size: {len(test_loader)}")
    # 训练循环
    for epoch in range(1, 51):
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
    

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
        if val_iou>best_iou:
            best_iou = val_iou
            print(f"New best IoU: {best_iou:.4f}, saving model...")
        # 保存最佳模型
        if val_iou == best_iou:
            torch.save(model.state_dict(), f"best_model.pth")
        
        #测试并且输出结果
        # if epoch % 10 == 0:
        model.eval()
        test_loss, test_iou = validate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f} | Test IoU: {test_iou:.4f}")




if __name__ == '__main__':
    main()