import random
from typing import List, Tuple

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
slide = openslide.OpenSlide(wsi_path)
sampling_strategy = "grid"  # or "random" or "tissue"
patch_size = 256

if sampling_strategy == "grid":
    coords = grid_sampling(best_dims, patch_size, stride=256)
elif sampling_strategy == "random":
    coords = random_sampling(best_dims, patch_size, n_patches=100)
elif sampling_strategy == "tissue":
    coords = tissue_sampling(slide, best_dims, patch_size, n_patches=100)