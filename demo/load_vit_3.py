import timm
from PIL import Image
from torchvision import transforms
import torch
import os

# assert "HF_TOKEN" in os.environ, "Please set the HF_TOKEN environment variable to your Hugging Face API token"
os.environ["HF_TOKEN"] = "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"

tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False) 
# print(tile_encoder)
model_path="/home/wcy/下载/pytorch_model.bin" 
# state_dict=torch.load(model_path)["model"]
# print(state_dict.keys())

tile_encoder.load_state_dict(torch.load(model_path))
# print(tile_encoder)
# print(next(tile_encoder.parameters()).device)
print("param #", sum(p.numel() for p in tile_encoder.parameters()))

transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

img_path = "images/prov_normal_000_1.png"
org_img=transform(Image.open(img_path).convert("RGB")).unsqueeze(0)
print(org_img.shape)
sample_input = transform(Image.open(img_path).convert("RGB")).unsqueeze(0)

print("Sample input:", sample_input.shape)

with torch.no_grad():
    output = tile_encoder(sample_input).squeeze()
    print("Model output:", output.shape)
    print(output)

expected_output = torch.load("images/prov_normal_000_1.pt")
print("Expected output:", expected_output.shape)
print(expected_output)

assert torch.allclose(output, expected_output, atol=1e-2)