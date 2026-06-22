import logging
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "final-model-vgg16.pth"
# ImageNet normalisation
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])
_model  = None
_device = None


def _build():
    """Build VGG16 with the slim classifier matching your trained weights."""
    model = tv_models.vgg16(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 128),
        nn.ReLU(inplace=False),   
        nn.Dropout(0.5),
        nn.Linear(128, 2),
    )
    return model


def load_model():
    global _model, _device

    if _model is not None:
        return _model, _device

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading model → %s on %s", MODEL_PATH.name, _device)

    model = _build()
    state = torch.load(MODEL_PATH, map_location=_device, weights_only=True)

    # Accept both bare state dicts and checkpoint dicts
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)

    # Disable ALL inplace ops — prevents Grad-CAM backward hook conflicts
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

    model.to(_device).eval()
    _model = model

    if _device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()

    return _model, _device
