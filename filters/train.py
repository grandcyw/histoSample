import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import openslide
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


from wsiLoader_level2_mixed import WSIDataset,create_datasets
from models.simplecnn import SimpleCNN
from models.densenet import DenseNet121
from models.unet import UNet

# from models.resnet import ResNet50
# from models.vgg import VGG16
# from models.efficientnet import EfficientNetB0
# from models.inceptionv3 import InceptionV3
# from models.mobilenet import MobileNetV2

class GleasonGradeCriterion(nn.Module):
    def __init__(self, grade_weights=None):
        super(GleasonGradeCriterion, self).__init__()
        # Default weights for grade 3, 4, 5 patterns
        # These weights can be adjusted based on clinical importance
        self.grade_weights = grade_weights if grade_weights is not None else {
            3: 0.3,
            4: 0.5,
            5: 0.7
        }
        
    def forward(self, predictions, masks):
        """
        Args:
            predictions: Tensor of shape [batch_size, 1] (predicted grades 1-6)
            masks: Tensor of shape [batch_size, 1, H, W, 1] (values 3,4,5 for Gleason patterns)
        
        Returns:
            loss: MSE loss between predicted grade and computed grade from mask
        """
        batch_size = masks.shape[0]
        
        # Flatten spatial dimensions (H,W)
        flat_masks = masks.view(batch_size, -1)  # [batch_size, H*W]
        
        # Compute proportions of each grade pattern
        grade_props = []
        for grade in [3, 4, 5]:
            count = (flat_masks == grade).float().sum(dim=1)  # [batch_size]
            prop = count / flat_masks.shape[1]  # proportion
            grade_props.append(prop)
        
        # Convert to tensor [batch_size, 3]
        grade_props = torch.stack(grade_props, dim=1)
        
        # Compute weighted score (higher weights for more aggressive patterns)
        weights = torch.tensor([self.grade_weights[3], 
                              self.grade_weights[4], 
                              self.grade_weights[5]], 
                             device=masks.device)
        weighted_scores = grade_props * weights.unsqueeze(0)
        
        # Compute final grade (1-6 scale)
        total_score = weighted_scores.sum(dim=1)  # [batch_size]
        target_grades = (total_score * 5 + 1).clamp(1, 6)  # map [0,1] to [1,6]
        
        # Compute MSE loss
        loss = nn.functional.mse_loss(predictions.squeeze(), target_grades)
        
        return loss


def create_loader(data_dir = "data/wsi/train", mask_dir = "data/wsi/labels", patch_size=256,level=2,batch_size=4, shuffle=True):
    # data_dir = "data/wsi/train"
    # mask_dir = "data/wsi/labels"
    patch_size = 256
    level = 2
    batch_size = 4
    
    # Create datasets with 80% train, 10% val, 10% test split
    train_datasets, val_datasets, test_datasets = create_datasets(
        data_dir, mask_dir, patch_size, level, val_ratio=0.1, test_ratio=0.1
    )
    
    # Since we have multiple datasets, we can either:
    # 1. Use them separately (train on each sequentially)
    # 2. Combine them using torch.utils.data.ConcatDataset
    
    # Option 1: Process each WSI dataset separately
    # print("\nProcessing train datasets:")
    # for i, dataset in enumerate(train_datasets[:3]):  # Just show first 3 for demo
    #     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #     print(f"\nTraining on WSI {i+1}/{len(train_datasets)}")
    #     for batch_idx, (patches, masks) in enumerate(loader):
    #         print(f"Batch {batch_idx+1}: Patches {patches.shape}, Masks {masks.shape}")
            # if batch_idx > 2:  # Just show 3 batches per WSI
            #     break
    
    # Option 2: Combine all train datasets
    from torch.utils.data import ConcatDataset
    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets)
    combined_test = ConcatDataset(test_datasets)
    
    train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(combined_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(combined_test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
    
    # print("\nTraining on combined dataset:")
    # for i, (patches, masks) in enumerate(train_loader):
    #     print(f"Batch {i+1}: Patches {patches.shape}, Masks {masks.shape}")

# 示例：加载WSI和mask
def train():
    import pandas as pd
    from collections import defaultdict

    # Load PANDA.csv with ground truth labels
    panda_df = pd.read_csv('PANDA.csv')
    # Create a dictionary mapping WSI filenames to ISUP grades
    wsi_to_grade = dict(zip(panda_df['image_id'], panda_df['isup_grade']))

    def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
        model.train()
        
        # Dictionary to store predictions by WSI
        wsi_predictions = defaultdict(list)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for i, (patches, masks, wsi_ids) in enumerate(train_loader):  # Modified to include WSI IDs
                patches = patches.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(patches)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Convert model outputs to predicted grades (assuming 6-class classification)
                _, preds = torch.max(outputs, 1)
                
                # Aggregate predictions by WSI
                for wsi_id, pred in zip(wsi_ids, preds):
                    wsi_predictions[wsi_id].append(pred.item())
                
                if i % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
            # Calculate WSI-level accuracy
            correct = 0
            total = 0
            for wsi_id, preds in wsi_predictions.items():
                # Get most frequent prediction for this WSI
                pred_grade = max(set(preds), key=preds.count)
                true_grade = wsi_to_grade.get(wsi_id, None)
                
                if true_grade is not None:
                    total += 1
                    if pred_grade == true_grade:
                        correct += 1
            
            accuracy = correct / total if total > 0 else 0
            avg_loss = epoch_loss / len(train_loader)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
            
            # Clear predictions for next epoch
            wsi_predictions.clear()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wsi_path = "demo/706a5789a3517393a583829512a1fb8d.tiff"
    mask_path = "demo/706a5789a3517393a583829512a1fb8d_mask.tiff"
    level = 2  # 使用WSI的最高分辨率层级
    dataset = WSIDataset(wsi_path, mask_path, patch_size=256, level=level)
    print(f"Dataset size: {len(dataset)} patches")
    for level_i in range(dataset.level_count):
        print(f"Level {level_i} WSI dimensions: {dataset.wsi.level_dimensions[level_i]}")
        print(f"Level {level_i} Mask dimensions: {dataset.mask.level_dimensions[level_i]}")
    print(f"WSI dimensions: {dataset.wsi.level_dimensions[level]}")
    print(f"Mask dimensions: {dataset.mask.level_dimensions[level]}")
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    train_loader, val_loader, test_loader = create_loader(data_dir="data/wsi/train", mask_dir="data/wsi/labels", patch_size=256, level=2, batch_size=4, shuffle=True)
    # for i, (patches, masks) in enumerate(dataloader):
    #     if patches is None or masks is None:
    #         continue
    #     print(f"Batch {i+1}:")
    #     print(f"Patches shape: {patches.shape}, Masks shape: {masks.shape}")
    #     # if i == 0:  # 只可视化第一个batch
    #     dataset.visualize(i)
    #     if i>5:
    #         break


    
    model= SimpleCNN()
    model.to(device)
    # criterion = nn.BCELoss()
    criterion = GleasonGradeCriterion()  # 使用自定义的Gleason分级损失函数
    # ValueError: Using a target size (torch.Size([4, 1, 256, 256, 1])) that is different to the input size (torch.Size([4, 1])) is deprecated. Please ensure they have the same size.
    # 需要将256*256的mask转换为1维, Gleason分级3\4\5级占比越高，最后分类1~6级约接近6级
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        # 训练模型
        for i, (patches, masks) in enumerate(train_loader):
            patches = patches.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')

        if epoch%2 == 0:    
            # 测试模型
            model.eval()
            with torch.no_grad():
                total_loss = 0.0
                for i, (patches, masks) in enumerate(val_loader):
                    patches = patches.to(device)
                    masks = masks.to(device)

                    outputs = model(patches)
                    loss = criterion(outputs, masks)
                    total_loss += loss.item()

                avg_loss = total_loss / len(val_loader)
                print(f'Validation Loss: {avg_loss:.4f}')
            model.train()

        # 保存模型
        torch.save(model.state_dict(), f'filters/checkpoints/SimpleCNN_epoch_{epoch+1}_acc{avg_loss:.4f}.pth')
