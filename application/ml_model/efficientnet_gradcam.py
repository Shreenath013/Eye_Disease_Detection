import os

import cv2
import torch
import numpy as np

from PIL import Image

from torchvision import transforms

from .model_loader import model
from .gradcam_utils import GradCAM

from django.conf import settings

import logging

logger = logging.getLogger(__name__)


# ==============================
# DEVICE
# ==============================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==============================
# TRANSFORM
# ==============================

transform = transforms.Compose([

    transforms.Resize((300, 300)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),

])


# ==============================
# TARGET LAYER
# ==============================

target_layer = model.features[-1]

# ==============================
# GENERATE HEATMAP FUNCTION
# ==============================

def generate_gradcam(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    # ==============================
    # FORWARD PASS
    # ==============================

    outputs = model(input_tensor)

    pred_idx = outputs.argmax(dim=1).item()
    
    # ==============================
    # CREATE FRESH GRADCAM INSTANCE
    # ==============================

    cam = GradCAM(
        model,
        target_layer
    )

    # ==============================
    # GENERATE GRADCAM
    # ==============================

    try:

        grayscale_cam = cam.generate(
            input_tensor=input_tensor,
            class_idx=pred_idx
        )

    finally:

        cam.remove_hooks()

    # ==============================
    # ORIGINAL IMAGE
    # ==============================

    rgb_img = np.array(
        image.resize((300, 300))
    ) / 255.0

    # ==============================
    # HEATMAP
    # ==============================
    
    grayscale_cam = cv2.resize(
        grayscale_cam,
        (rgb_img.shape[1], rgb_img.shape[0])
    )

    heatmap = cv2.applyColorMap(
        np.uint8(255 * grayscale_cam),
        cv2.COLORMAP_JET
    )

    heatmap = np.float32(heatmap) / 255

    overlay = heatmap + rgb_img

    overlay = overlay / (overlay.max() + 1e-8)

    overlay = np.uint8(255 * overlay)

    # ==============================
    # SAVE HEATMAP
    # ==============================

    output_dir = os.path.join(
        settings.MEDIA_ROOT,
        "gradcam_outputs"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    filename = os.path.basename(image_path)

    save_path = os.path.join(
        output_dir,
        f"gradcam_{filename}"
    )

    cv2.imwrite(
        save_path,
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )
    )

    logger.info(f"GradCAM saved: {save_path}")

    return save_path