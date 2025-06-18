# sudo apt update
# sudo apt install fonts-wqy-zenhei
# fc-list | grep "WenQuanYi Zen Hei"
# rm -rf ~/.cache/matplotlib
# python sampling/sample_visualize_with_args.py --wsi_path demo/706a5789a3517393a583829512a1fb8d.tiff --output_dir ./results

import os
import argparse
import openslide
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
# import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt
from matplotlib import font_manager
import sys

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 使用文泉驿正黑
# plt.title("中文标题测试")
# plt.savefig("test.png", bbox_inches='tight', dpi=300)

# sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="WSI 切片采样与可视化")
    parser.add_argument("--wsi_path", type=str, required=True, help="WSI 文件路径（.svs/.tiff）")
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录（默认：./output）")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载 WSI
    slide = openslide.OpenSlide(args.wsi_path)
    width, height = slide.dimensions

    # 2. 生成缩略图
    thumbnail = slide.get_thumbnail((2000, 2000))
    thumbnail = np.array(thumbnail)

    # 3. 定义选中的子图坐标（示例）
    selected_patches = [
        (1000, 2000, 512, 512),
        (3000, 4000, 512, 512),
        (5000, 6000, 512, 512),
    ]

    # 4. 在缩略图上标出选中的子图
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(thumbnail)

    scale_x = thumbnail.shape[1] / width
    scale_y = thumbnail.shape[0] / height

    for (x, y, w, h) in selected_patches:
        rect_x = x * scale_x
        rect_y = y * scale_y
        rect_w = w * scale_x
        rect_h = h * scale_y

        rect = Rectangle(
            (rect_x, rect_y), rect_w, rect_h,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(rect_x, rect_y, f"({x}, {y})", color="red", fontsize=10)

    plt.title("WSI 缩略图（红框标出选中的组织子图）")
    plt.axis("off")
    plt.savefig(os.path.join(args.output_dir, "wsi_with_selected_patches.png"), bbox_inches="tight", dpi=300)
    plt.close()

    # 5. 提取选中的子图（原分辨率）
    for idx, (x, y, w, h) in enumerate(selected_patches):
        patch = slide.read_region((x, y), 0, (w, h))
        patch = patch.convert("RGB")
        patch.save(os.path.join(args.output_dir, f"selected_patch_{idx}.png"))

if __name__ == "__main__":
    main()