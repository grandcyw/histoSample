import random
from typing import List, Tuple
import openslide
from find_best_level_1 import find_best_level

def grid_sampling(dims: Tuple[int, int], patch_size: int, stride: int) -> List[Tuple[int, int]]:
    """Grid sampling strategy"""
    coords = []
    for x in range(0, dims[0] - patch_size, stride):
        for y in range(0, dims[1] - patch_size, stride):
            coords.append((x, y))
    return coords

def random_sampling(dims: Tuple[int, int], patch_size: int, n_patches: int) -> List[Tuple[int, int]]:
    """Random sampling strategy"""
    coords = []
    for _ in range(n_patches):
        x = random.randint(0, dims[0] - patch_size)
        y = random.randint(0, dims[1] - patch_size)
        coords.append((x, y))
    return coords

def tissue_sampling(slide, dims: Tuple[int, int], patch_size: int, n_patches: int) -> List[Tuple[int, int]]:
    """Tissue-aware sampling (simplified)"""
    # (In practice, use Otsu thresholding or ML-based tissue detection)
    thumbnail = slide.get_thumbnail((dims[0] // 10, dims[1] // 10))
    thumbnail_gray = thumbnail.convert('L')
    coords = []
    while len(coords) < n_patches:
        x = random.randint(0, dims[0] - patch_size)
        y = random.randint(0, dims[1] - patch_size)
        # Check if patch is in tissue (simplified)
        if thumbnail_gray.getpixel((x // 10, y // 10)) < 240:  # Not white background
            coords.append((x, y))
    return coords


# Example usage
wsi_path = "demo/706a5789a3517393a583829512a1fb8d.tiff"
slide = openslide.OpenSlide(wsi_path)
best_level, best_mpp, best_dims = find_best_level(wsi_path)
sampling_strategy = "grid"  # or "random" or "tissue"
patch_size = 256

coords = []

if sampling_strategy == "grid":
    coords = grid_sampling(best_dims, patch_size, stride=256)
elif sampling_strategy == "random":
    coords = random_sampling(best_dims, patch_size, n_patches=100)
elif sampling_strategy == "tissue":
    coords = tissue_sampling(slide, best_dims, patch_size, n_patches=100)

print(f"Sampling strategy: {sampling_strategy}")
print(f"Number of patches sampled: {len(coords)}")
# for coord in coords:
#     print(f"Patch coordinates: {coord}")
print("Sampled patch coordinates:", coords[:5])  # Print first 5 for brevity
slide.close()