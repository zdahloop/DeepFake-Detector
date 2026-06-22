# DeepScan — Deepfake Image Detection System

A professional Flask web application for detecting AI-generated / deepfake faces
using a fine-tuned VGG16 model, MTCNN face detection, and Grad-CAM explainability.

---

## Project Structure

```
deepfake_detector/
├── app.py              # Flask routes and application entry point
├── model.py            # VGG16 loader, MTCNN face detection, prediction logic
├── gradcam.py          # Grad-CAM implementation and visualisation helpers
├── requirements.txt    # Python dependencies
├── model.pt            # ← Place your trained VGG16 weights here
├── templates/
│   ├── index.html      # Upload page
│   └── result.html     # Results + Grad-CAM visualisation page
└── static/
    └── style.css       # Dark forensic-lab UI stylesheet
```

---

## Quick Start

### 1. Install dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install all packages
pip install -r requirements.txt

# CPU-only PyTorch (no GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Add your model weights

Place your trained VGG16 state dict at the project root:

```
deepfake_detector/model.pt
```

The file can be:
- A bare `state_dict` saved with `torch.save(model.state_dict(), 'model.pt')`
- A checkpoint dict with key `"model_state_dict"`:
  `torch.save({"model_state_dict": model.state_dict(), ...}, 'model.pt')`

**Expected architecture**: VGG16 with `classifier[6]` replaced by `Linear(4096, 2)`.
Class mapping: `{0: "REAL", 1: "FAKE"}` — update `CLASS_LABELS` in `model.py` if different.

### 3. Run the app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## How It Works

```
Upload image
    │
    ▼
MTCNN face detection
    │ face found → crop + margin
    │ no face    → use full image
    ▼
Resize to 224×224
Normalize with ImageNet mean/std
    │
    ▼
VGG16 forward pass
    │
    ▼
Softmax → P(REAL), P(FAKE)
    │
    ▼
Grad-CAM @ features.28 (last conv block)
  • Forward hook captures activations
  • Backward pass on predicted class score
  • Pool gradients → weight activations
  • Upsample → JET colormap → overlay
    │
    ▼
Result page:
  • Verdict (REAL / FAKE)
  • Confidence % (animated gauge)
  • Probability bars
  • Three Grad-CAM panels: original / heatmap / overlay
  • Model metadata table
```

---

## Configuration

| Setting | Location | Default |
|---------|----------|---------|
| Model path | `model.py → MODEL_PATH` | `./model.pt` |
| Class labels | `model.py → CLASS_LABELS` | `{0:"REAL", 1:"FAKE"}` |
| Grad-CAM layer | `app.py → _run_gradcam()` | `features.28` |
| Display image size | `app.py → _run_gradcam()` | `400×400 px` |
| Heatmap opacity | `gradcam.py → overlay_heatmap()` | `α = 0.45` |
| Max upload size | `app.py` | `16 MB` |
| Port | `app.py` | `5000` |

---

## Privacy & Safety

- **No files saved to disk.** Images are processed entirely in memory using `BytesIO`.
- Results are stored in a server-side session (encrypted with a random key generated at startup).
- Session data is cleared when you navigate to a new analysis.

---

## Training Your Own Model

If you need to train a VGG16 deepfake detector:

```python
import torch
import torchvision.models as models
import torch.nn as nn

model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model.classifier[6] = nn.Linear(4096, 2)  # binary: real vs fake

# ... train on your dataset ...

torch.save(model.state_dict(), "model.pt")
```

Popular datasets: FaceForensics++, DFDC (Deepfake Detection Challenge), Celeb-DF.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `FileNotFoundError: model.pt` | Place your weights file at the project root |
| `RuntimeError: size mismatch` | Your model has a different head size — update `_build_vgg16()` in `model.py` |
| MTCNN not found | Run `pip install facenet-pytorch`; the app falls back to full image without it |
| CUDA out of memory | Add `torch.cuda.empty_cache()` or use CPU mode |
| Slow first request | The model is pre-warmed at startup; subsequent requests are faster |
