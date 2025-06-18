import openslide
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# 1. 加载 WSI
slide = openslide.OpenSlide("demo/706a5789a3517393a583829512a1fb8d.tiff")
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
    ax.text(rect_x, rect_y, f"({x}, {y})", color="yellow", fontsize=10)

plt.title("WSI 缩略图（红框标出选中的组织子图）")
plt.axis("off")
plt.savefig("wsi_with_selected_patches.png", bbox_inches="tight", dpi=300)
plt.show()

# 5. 提取选中的子图（原分辨率）
for idx, (x, y, w, h) in enumerate(selected_patches):
    patch = slide.read_region((x, y), 0, (w, h))
    patch = patch.convert("RGB")
    patch.save(f"selected_patch_{idx}.png")