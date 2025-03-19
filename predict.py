from torchvision import models, transforms
from torch import load, no_grad
import torch
from PIL import Image
import torch.nn as nn

def load_model(model_path: str, device="cpu"):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  
    model.load_state_dict(load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model

model = load_model("defect_model.pth")
#print(model)

def process_image(image_path):
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")

    return val_transforms(image).unsqueeze(0)


image = process_image("dataset/1.jpg")

def predict(model, image_path, device="cpu"):
    image = process_image(image_path).to(device)
    with no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

        return predicted.item()

print(predict(model, "dataset/train/defect/20200707_085703 - Copy.jpg"))