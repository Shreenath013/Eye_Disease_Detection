import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    """
    GradCAM implementation for EfficientNet-B3 based retinal image classification and explainable AI visualization.
    """
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None
        self.gradients    = None
        self._handles     = []
        self._register()

    def _register(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._handles.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self._handles.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        out = self.model(input_tensor)
        out[:, class_idx].sum().backward()   # sum() handles batch > 1 safely

        grads   = self.gradients                           # (1, C, H, W)
        weights = grads.mean(dim=(2, 3), keepdim=True)    # global avg pool
        cam     = (weights * self.activations).sum(dim=1) # (1, H, W)
        cam     = F.relu(cam)
        cam = cam.squeeze(0).squeeze(0).cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam

    def remove_hooks(self):
        """Call after you're done to free memory."""
        for h in self._handles:
            h.remove()
        self._handles = []