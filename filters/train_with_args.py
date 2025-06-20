import argparse
import torch
import torch.nn as nn
from torchvision import models
from train import create_loader, validate

def parse_args():
    parser = argparse.ArgumentParser(description='Gleason Grading Segmentation Training')
    
    # 模型选择
    parser.add_argument('--model', type=str, default='unet',
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
    parser.add_argument('--image-dir', type=str, required=True,
                       help='Path to training images directory')
    parser.add_argument('--mask-dir', type=str, required=True,
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

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 准备数据
    # dataset = SegmentationDataset(args.image_dir, args.mask_dir)
    # train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    train_loader, val_loader, test_loader = create_loader(data_dir="data/wsi/train", mask_dir="data/wsi/labels", patch_size=256, level=2, batch_size=4, shuffle=True)

    
    # 2. 初始化模型
    model = get_model(args).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 3. 训练循环
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images = images.permute(0, 3, 1, 2).float().to(device)
            masks = masks.unsqueeze(1).float().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{args.epochs} | '
              f'Train Loss: {train_loss/len(train_loader):.4f} | '
              f'Val Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        if args.save_best and val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'{args.output_dir}/best_model.pth')
            print('Saved best model!')

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.permute(0, 3, 1, 2).float().to(device)
            masks = masks.unsqueeze(1).float().to(device)
            
            outputs = model(images)
            val_loss += criterion(outputs, masks).item()
    
    model.train()
    return val_loss / len(val_loader)

if __name__ == '__main__':
    main()