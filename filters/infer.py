def predict_and_save(model, wsi_path, output_path, patch_size=256, level=0, threshold=0.5):
    wsi = openslide.OpenSlide(wsi_path)
    wsi_width, wsi_height = wsi.level_dimensions[level]
    output_mask = np.zeros((wsi_height, wsi_width), dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        for y in range(0, wsi_height, patch_size):
            for x in range(0, wsi_width, patch_size):
                patch = wsi.read_region((x, y), level, (patch_size, patch_size))
                patch = patch.convert('RGB')
                patch = np.array(patch) / 255.0
                patch = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float().to(device)

                output = model(patch)
                output = (output.squeeze().cpu().numpy() > threshold).astype(np.uint8) * 255

                # 将预测结果拼接到完整mask
                output_mask[y:y+patch_size, x:x+patch_size] = output

    # 保存结果
    Image.fromarray(output_mask).save(output_path)

# 示例推理
predict_and_save(model, "path/to/test_wsi.svs", "path/to/output_mask.png")

def visualize_results(wsi_path, mask_path, output_path):
    wsi = openslide.OpenSlide(wsi_path)
    thumbnail = wsi.get_thumbnail((1024, 1024))
    mask = Image.open(mask_path)
    output = Image.open(output_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(thumbnail)
    axes[0].set_title('WSI Thumbnail')
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[2].imshow(output, cmap='gray')
    axes[2].set_title('Predicted Mask')
    plt.show()

visualize_results("path/to/test_wsi.svs", "path/to/mask.png", "path/to/output_mask.png")