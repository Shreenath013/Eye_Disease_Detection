import os

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torchvision import transforms
from PIL import Image

from .model_loader import model

import logging

logger = logging.getLogger(__name__)


# ==============================
# DEVICE CONFIGURATION
# ==============================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

logger.info(f"Using device: {device}")


# ==============================
# BASE DIRECTORY
# ==============================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ==============================
# CLASS LABELS
# ==============================

class_names = [
    "myopia",
    "normal",
    "pathological_myopia"
]


# ==============================
# IMAGE TRANSFORM
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
# PREDICTION FUNCTION
# ==============================

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(
        device,
        memory_format=torch.channels_last
    )

    with torch.no_grad():

        if device.type == "cuda":

            with autocast(device_type="cuda"):

                outputs = model(input_tensor)

        else:

            outputs = model(input_tensor)
        
        del input_tensor
        
        probs = F.softmax(outputs, dim=1).squeeze()

    # ==============================
    # CALIBRATED NORMAL THRESHOLD
    # ==============================

    NORMAL_IDX = 1

    NORMAL_THRESHOLD = 0.30

    probs_list = probs.tolist()

    normal_prob = probs_list[NORMAL_IDX]

    if normal_prob >= NORMAL_THRESHOLD:

        pred_idx = NORMAL_IDX

    else:

        pred_idx = torch.argmax(probs).item()

    prediction = class_names[pred_idx]

    confidence = probs_list[pred_idx] * 100

    # ==============================
    # TERMINAL OUTPUT
    # ==============================

    print("\n===== PREDICTION RESULT =====")

    print(f"Image           : {image_path}")

    print(f"Prediction      : {prediction.upper()}")

    print(f"Confidence      : {confidence:.2f}%")

    print(
        f"Normal threshold: {NORMAL_THRESHOLD} "
        f"(normal prob={normal_prob * 100:.2f}%)"
    )

    print("\nAll class probabilities:")

    for cls, prob in zip(class_names, probs_list):

        bar = "█" * int(prob * 30)

        marker = (
            " ← predicted"
            if cls == prediction
            else ""
        )

        print(
            f"  {cls:25s}: "
            f"{prob * 100:5.2f}%  "
            f"{bar}{marker}"
        )

    probabilities = {

        "myopia": probs_list[0] * 100,

        "normal": probs_list[1] * 100,

        "pathological_myopia": probs_list[2] * 100,

    }
    
    if device.type == "cpu":
        import gc
        gc.collect()

    return prediction, confidence, probabilities


# ==============================
# DIRECT TESTING
# ==============================

if __name__ == "__main__":

    image_path = os.path.join(
        BASE_DIR,
        "demo.jpg"
    )

    predict_image(image_path)