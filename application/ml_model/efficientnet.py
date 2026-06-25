import torch
import torch.nn as nn
from torchvision import models


def get_efficientnet(num_classes=3):
    """
    # EfficientNet-B3: ~12M parameters
    # Model input size used in this project: 300×300
    Chosen over B0 (5.3M, 224px) for better feature resolution on the
    subtle normal/myopia fundus boundary.
    Chosen over B4 (19M, 300px) because B4 OOMs at batch 32 on RTX 3050 6GB.

    Head architecture: Dropout → Linear → BN → SiLU → Dropout → Linear
    # EfficientNet-B3 classifier input features = 1536
    # (EfficientNet-B0 uses 1280 features)
    """
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

    # Phase-1: freeze entire backbone
    for param in model.parameters():
        param.requires_grad = False

    # Keep BN frozen in eval mode — stable ImageNet stats
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.track_running_stats = True

    in_features = model.classifier[1].in_features  # 1536 for B3

    # Stronger regularisation in head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )

    return model


def unfreeze_layers(model, layers=("features.5", "features.6", "features.7", "features.8")):
    """
    Unfreeze specified backbone layers for Phase-2 fine-tuning.

    EfficientNet-B3 feature blocks (9 total: 0=stem, 1-8=MBConv):
      features.0  — stem conv
      features.1  — MBConv block 1  (stride 1)
      features.2  — MBConv block 2
      features.3  — MBConv block 3
      features.4  — MBConv block 4
      features.5  — MBConv block 5  ← unfreeze from here
      features.6  — MBConv block 6
      features.7  — MBConv block 7
      features.8  — top conv + BN

    B3 has more blocks than B0 (which only had 6), so we unfreeze from
    block 5 onward (4 blocks vs 3 in B0) to get sufficient fine-tuning
    depth while keeping early texture/edge features frozen.
    """
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers):
            param.requires_grad = True

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            if any(layer in name for layer in layers):
                module.train()
                module.track_running_stats = True

    unfrozen = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Unfrozen: {len(unfrozen)} tensors from layers {layers}")
    return model
