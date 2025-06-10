from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import cv2
import os

def generate_heatmap(model, image_path, input_tensor=None, target_class=0):
    if input_tensor is None:
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([...])
        input_tensor = transform(image).unsqueeze(0)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    target_layer = model.layer4[-1]

    cam = GradCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=[ClassifierOutputTarget(target_class)]
    )
    grayscale_cam = grayscale_cam[0, :]

    rgb_img = np.array(image.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join("media", f"{name}_heatmap.png")
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print("[Grad-CAM] Сохранил heatmap:", output_path)
    output_path = output_path.replace("\\", "/")
    return output_path