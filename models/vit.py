import torch
from torchvision import transforms
from transformers import ViTModel, ViTConfig

# Load pretrained ViT
vit_config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
vit_model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(slide, coords: List[Tuple[int, int]], patch_size: int):
    features = []
    for (x, y) in coords:
        patch = slide.read_region((x, y), best_level, (patch_size, patch_size))
        patch = patch.convert("RGB")
        patch = transform(patch).unsqueeze(0)  # (1, 3, 224, 224)
        with torch.no_grad():
            outputs = vit_model(patch)
            features.append(outputs.last_hidden_state.mean(dim=1))  # Pooled features
    return torch.cat(features, dim=0)  # (N, 768)

features = extract_features(slide, coords, patch_size)