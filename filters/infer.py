# def predict_and_save(model, wsi_path, output_path, patch_size=256, level=0, threshold=0.5):
#     wsi = openslide.OpenSlide(wsi_path)
#     wsi_width, wsi_height = wsi.level_dimensions[level]
#     output_mask = np.zeros((wsi_height, wsi_width), dtype=np.uint8)

#     model.eval()
#     with torch.no_grad():
#         for y in range(0, wsi_height, patch_size):
#             for x in range(0, wsi_width, patch_size):
#                 patch = wsi.read_region((x, y), level, (patch_size, patch_size))
#                 patch = patch.convert('RGB')
#                 patch = np.array(patch) / 255.0
#                 patch = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float().to(device)

#                 output = model(patch)
#                 output = (output.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255

#                 # 将预测结果拼接到完整mask
#                 output_mask[y:y+patch_size, x:x+patch_size] = output

#     # 保存结果
#     Image.fromarray(output_mask).save(output_path)

# # 示例推理
# predict_and_save(model, "path/to/test_wsi.svs", "path/to/output_mask.png")

# def visualize_results(wsi_path, mask_path, output_path):
#     wsi = openslide.OpenSlide(wsi_path)
#     thumbnail = wsi.get_thumbnail((1024, 1024))
#     mask = Image.open(mask_path)
#     output = Image.open(output_path)

#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     axes[0].imshow(thumbnail)
#     axes[0].set_title('WSI Thumbnail')
#     axes[1].imshow(mask, cmap='gray')
#     axes[1].set_title('Ground Truth Mask')
#     axes[2].imshow(output, cmap='gray')
#     axes[2].set_title('Predicted Mask')
#     plt.show()

# visualize_results("path/to/test_wsi.svs", "path/to/mask.png", "path/to/output_mask.png")


import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

def visualize_prediction(image, mask, pred):
    """
    image: [3,H,W] 原始图像
    mask: [H,W] 真实掩码
    pred: [H,W] 预测掩码 (经过sigmoid)
    """
    plt.figure(figsize=(15,5))
    
    plt.subplot(131)
    plt.imshow(image.permute(1,2,0).cpu().numpy())
    plt.savefig('input_image.png')
    
    plt.title("Input Image")
    
    plt.subplot(132)
    mask = mask.squeeze()  # 确保掩码是2D的
    plt.imshow(mask.cpu().numpy(), cmap='gray')
    plt.savefig('ground_truth.png')

    plt.title("Ground Truth")
    
    plt.subplot(133)
    plt.imshow(torch.sigmoid(pred).cpu().numpy() > 0.5, cmap='gray')
    plt.title("Prediction")
    
    plt.savefig('prediction.png')

def get_model():
    # 假设你已经定义了一个模型类，比如UNet
    from models.unet import UNet  # 替换为你的模型导入路径
    model = UNet(in_channels=3, out_channels=1)  # 根据实际情况调整参数
    torch.load(f'filters/checkpoints/256*256/best_model.pth', map_location='cpu')  # 替换为你的模型路径
    model.eval()  # 设置为评估模式
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # 将模型移动到设备上
    return model

if __name__ == "__main__":


    model = get_model()
    # 使用示例
    from filters.train import create_loader  # 替换为你的数据集加载函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = create_loader(
        data_dir="data/wsi/train",
        mask_dir="data/wsi/labels",
        patch_size=256,
        level=2,
        batch_size=4,
        shuffle=True
    )
    with torch.no_grad():
        image, mask = next(iter(val_loader))
        pred = model(image.to(device))[0]
        visualize_prediction(image[0], mask[0], pred[0])


