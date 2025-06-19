import os
import random
from pathlib import Path

def sample_and_link(source_dir="raw_data", 
                   target_dir="data", 
                   sample_size=1000, 
                   total_files=10000,
                   label_dir="labels",
                   ext=".tiff"):
    """
    从源目录采样TIFF文件并创建软链接到目标目录
    
    参数:
        source_dir: 原始数据目录路径
        target_dir: 目标目录路径
        sample_size: 需要采样的文件数 (1000)
        total_files: 总文件数 (10000)
        ext: 文件扩展名 (默认为.tif)
    """
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取所有TIFF文件路径
    all_files = list(Path(source_dir).rglob(f"*{ext}"))
    assert len(all_files) >= total_files, f"实际文件数 {len(all_files)} < 总文件数 {total_files}"
    
    # 随机采样
    sampled_files = random.sample(all_files, sample_size)
    
    print(f"Sampled {len(sampled_files)} files from {source_dir}")
    print(f"Sampled files: {[str(file) for file in sampled_files[:5]]}...")  # 打印前5个采样文件
    print(f"Creating symlinks in {target_dir}...")

    sampled_labels=list(map(lambda x: Path(str(x).replace("train_images", "train_label_masks").replace(".tiff","_mask.tiff")), sampled_files))

    # 创建软链接
    success_count = 0
    for src_path in sampled_files:
        # 保持原始相对路径结构
        rel_path = src_path.relative_to(source_dir)
        dst_path = Path(target_dir) / rel_path
        
        # 确保目标父目录存在
        dst_path.parent.mkdir(parents=True, exist_ok=True)        
        try:
            # 创建软链接（跨平台兼容）
            if os.path.exists(dst_path):
                os.remove(dst_path)
            os.symlink(src_path.resolve(), dst_path)
            success_count += 1
        except OSError as e:
            print(f"Failed to link {src_path}: {e}")
    for src_path in sampled_labels:
        # 保持原始相对路径结构
        rel_path = src_path.relative_to("/data/PANDA_grading/train_label_masks")
        dst_path = Path(label_dir) / rel_path
        
        # 确保目标父目录存在
        dst_path.parent.mkdir(parents=True, exist_ok=True)        
        try:
            # 创建软链接（跨平台兼容）
            if os.path.exists(dst_path):
                os.remove(dst_path)
            os.symlink(src_path.resolve(), dst_path)
            success_count += 1
        except OSError as e:
            print(f"Failed to link {src_path}: {e}")
    
    print(f"Successfully created {success_count}/{sample_size} symlinks in {target_dir}")

if __name__ == "__main__":
    sample_and_link(source_dir="/data/PANDA_grading/train_images/",
                   target_dir="./data/wsi/train",
                   sample_size=1000,
                   total_files=10000,label_dir="data/wsi/labels")
    # sample_and_link(source_dir="/data/PANDA_grading/train_label_masks/",
    #             target_dir="./data/wsi/labels",
    #             sample_size=1000,
    #             total_files=10000)