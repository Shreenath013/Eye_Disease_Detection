import os
import torch
from .efficientnet import (
    get_efficientnet,
    unfreeze_layers
)
import logging

logger = logging.getLogger(__name__)

# ==============================
# DEVICE CONFIGURATION
# ==============================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

logger.info("Using device: %s", device)


# ==============================
# MODEL PATH
# ==============================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_efficientnetb3_myopia.pth"
)


# ==============================
# LOAD MODEL
# ==============================

model = get_efficientnet(
    num_classes=3
)

model = unfreeze_layers(

    model,

    layers=(
        "features.5",
        "features.6",
        "features.7",
        "features.8",
    ),

)

MODEL_LOADED = False


# ==============================
# LOAD WEIGHTS
# ==============================

try:

    model.load_state_dict(

        torch.load(

            MODEL_PATH,

            map_location=device,

            weights_only=True

        )

    )

    model.to(device)

    model.eval()

    MODEL_LOADED = True

    logger.info(
        "EfficientNet-B3 model loaded successfully"
    )

except Exception as e:

    MODEL_LOADED = False

    raise RuntimeError(
        f"EfficientNet-B3 model failed to load: {e}"
    )