import openslide
import numpy as np

def find_best_level(wsi_path, target_mpp=0.5):
    slide = openslide.OpenSlide(wsi_path)
    # Get all level dimensions and MPPs
    levels = []
    for level in range(slide.level_count):
        dims = slide.level_dimensions[level]
        mpp_x = float(slide.properties.get('openslide.mpp-x', 0))
        mpp_y = float(slide.properties.get('openslide.mpp-y', 0))
        if mpp_x == 0 or mpp_y == 0:
            # Estimate MPP if not available
            base_dims = slide.level_dimensions[0]
            downsample = base_dims[0] / dims[0]
            mpp_x = mpp_y = 0.5 * downsample  # Assuming base MPP is 0.5
        mpp = (mpp_x + mpp_y) / 2
        levels.append((level, mpp, dims))
    
    # Find level with MPP closest to target_mpp
    best_level = min(levels, key=lambda x: abs(x[1] - target_mpp))
    return best_level[0], best_level[1], best_level[2]

wsi_path = "demo/706a5789a3517393a583829512a1fb8d.tiff"
best_level, best_mpp, best_dims = find_best_level(wsi_path)
print(f"Best level: {best_level}, MPP: {best_mpp}, Dimensions: {best_dims}")