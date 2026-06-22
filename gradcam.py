import cv2
import numpy as np
import torch

class GradCAM:
    def __init__(self, model, target_layer="features.28"):
        self.model = model
        self.model.eval()
        self._act  = None
        self._grad = None

        layer = dict(model.named_modules()).get(target_layer)
        if layer is None:
            raise ValueError(f"Layer '{target_layer}' not found.")

        self._fh = layer.register_forward_hook(self._fwd)
        self._bh = layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _, __, out):
        self._act = out.detach().clone()          
    def _bwd(self, _, __, grad_out):
        self._grad = grad_out[0].detach().clone() 

    def generate(self, tensor, class_idx=None):
        self.model.zero_grad()
        out = self.model(tensor)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        out[0, class_idx].backward()

        weights = self._grad.mean(dim=[0, 2, 3])
        cam = self._act[0].clone()
        for i, w in enumerate(weights):
            cam[i] *= w

        heatmap = cam.mean(0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap.astype(np.float32)

    def remove_hooks(self):
        self._fh.remove()
        self._bh.remove()


def apply_colormap(heatmap, size):
    h = np.uint8(255 * heatmap)
    h = cv2.resize(h, size)
    return cv2.applyColorMap(h, cv2.COLORMAP_JET)


def overlay_heatmap(orig, colored, alpha=0.45):
    return cv2.addWeighted(colored, alpha, orig, 1 - alpha, 0)
