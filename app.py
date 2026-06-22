import io
import time
import logging
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
import torch

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ALLOWED = {"jpg", "jpeg", "png", "webp"}

# ── Singletons (loaded once at startup) ──────────────────────────────────────
_mtcnn  = None
_device = None

def get_mtcnn():
    
    global _mtcnn, _device
    if _mtcnn is not None:
        return _mtcnn

    try:
        from facenet_pytorch import MTCNN
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _mtcnn  = MTCNN(
            keep_all      = False,   # return only the best face
            min_face_size = 40,      # minimum face size in pixels
            thresholds    = [0.6, 0.7, 0.7],  # P-Net / R-Net / O-Net thresholds
            device        = _device,
        )
        log.info("MTCNN loaded on %s", _device)
    except ImportError:
        log.warning("facenet-pytorch not installed — face detection unavailable.")
        _mtcnn = None

    return _mtcnn

# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED

def to_b64(bgr: np.ndarray) -> str:
    """Encode a BGR numpy image as a base-64 JPEG string."""
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode()

def detect_face(pil_img: Image.Image) -> tuple:
    mtcnn = get_mtcnn()
    if mtcnn is None:
        return None, None, (
            "Face detection is unavailable (facenet-pytorch not installed). "
            "Please run: pip install facenet-pytorch"
        )
    try:
        boxes, probs = mtcnn.detect(pil_img)
    except Exception as exc:
        log.exception("MTCNN detection error")
        return None, None, f"Face detection failed: {exc}"

    # No face found
    if boxes is None or len(boxes) == 0:
        return None, None, (
            "No face detected in this image. "
            "Please upload a clear portrait photo."
        )

    # Face found but confidence too low
    if probs[0] < 0.75:
        return None, None, (
            f"Face detected but confidence is too low ({probs[0]*100:.1f}%). "
            "Please use a clearer, well-lit portrait photo."
        )

    # ── Crop the face with a 15% padding margin ──────────────────────────────
    x1, y1, x2, y2 = [int(v) for v in boxes[0]]
    w, h = pil_img.size
    pad_x = int((x2 - x1) * 0.15)
    pad_y = int((y2 - y1) * 0.15)
    face_pil = pil_img.crop((
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    ))
    face_conf = round(float(probs[0]) * 100, 1)
    log.info("Face detected — box (%d,%d,%d,%d)  conf %.2f", x1, y1, x2, y2, probs[0])
    return face_pil, face_conf, None

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")
@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.time()

    # ── 1. Validate upload ────────────────────────────────────────────────────
    file = request.files.get("image")
    if not file or file.filename == "":
        return render_template("index.html", error="Please select an image file.")
    if not allowed(file.filename):
        return render_template("index.html", error="Only JPG, PNG and WebP images are supported.")

    raw = file.read()
    if not raw:
        return render_template("index.html", error="The uploaded file is empty.")

    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return render_template("index.html", error="Could not read the image. Please try another file.")

    orig_w, orig_h = pil_img.size
    fmt          = file.filename.rsplit(".", 1)[1].upper()
    file_size_kb = round(len(raw) / 1024, 1)

    # ── 2. MTCNN face detection (MANDATORY) ───────────────────────────────────
    
    face_img, face_conf, face_error = detect_face(pil_img)

    if face_error:
        log.warning("Face detection rejected: %s", face_error)
        return render_template("index.html", error=face_error)

    # face_img is now a clean PIL crop of the detected face
    face_w, face_h = face_img.size
    face_bgr       = cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR)
    log.info("Face crop size: %d × %d", face_w, face_h)

    # ── 3. Preprocess & predict (face crop only) ──────────────────────────────
    from model import load_model, preprocess
    model, device = load_model()

    # Preprocess the face crop → normalised 224×224 tensor
    tensor = preprocess(face_img).unsqueeze(0).to(device)
    tensor = tensor.clone().requires_grad_(True)   # required for Grad-CAM

    logits  = model(tensor)
    probs_t = torch.softmax(logits, dim=1)[0]
    cls       = probs_t.argmax().item()
    label     = "REAL" if cls == 0 else "FAKE"
    conf      = round(probs_t[cls].item() * 100, 1)
    prob_real = round(probs_t[0].item() * 100, 1)
    prob_fake = round(probs_t[1].item() * 100, 1)
    log.info("Prediction: %s  conf=%.1f%%  real=%.1f%%  fake=%.1f%%",
             label, conf, prob_real, prob_fake)

    # ── 4. Grad-CAM (face crop only) ──────────────────────────────────────────
    
    original_b64 = heatmap_b64 = overlay_b64 = None
    try:
        from gradcam import GradCAM, apply_colormap, overlay_heatmap
        gcam  = GradCAM(model, target_layer="features.28")
        hmap  = gcam.generate(tensor, class_idx=cls)
        gcam.remove_hooks()

        if device.type == "cuda":
            torch.cuda.empty_cache()
        # Resize face crop to display size — same crop the model used
        disp    = cv2.resize(face_bgr, (400, 400))
        colored = apply_colormap(hmap, (400, 400))
        blended = overlay_heatmap(disp, colored, alpha=0.45)

        original_b64 = to_b64(disp)
        heatmap_b64  = to_b64(colored)
        overlay_b64  = to_b64(blended)

        log.info("Grad-CAM generated on face crop (%d×%d → 400×400)", face_w, face_h)

    except Exception as exc:
        log.warning("Grad-CAM skipped: %s", exc)

    elapsed = round(time.time() - t0, 2)

    return render_template(
        "result.html",
        # verdict
        label     = label,
        conf      = conf,
        prob_real = prob_real,
        prob_fake = prob_fake,
        # grad-cam images (all face-crop based)
        original  = original_b64,
        heatmap   = heatmap_b64,
        overlay   = overlay_b64,
        # metadata
        filename     = file.filename,
        file_size_kb = file_size_kb,
        resolution   = f"{orig_w} × {orig_h}",
        face_res     = f"{face_w} × {face_h}",
        fmt          = fmt,
        color_space  = "RGB",
        face_detected = True,           
        face_conf    = face_conf,
        elapsed      = elapsed,
        device       = str(device),
        model_name   = "VGG16",
        gradcam_layer = "features.28",
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from model import load_model

    log.info("Pre-loading model…")
    try:
        load_model()
        log.info("Model ready.")
    except FileNotFoundError as exc:
        log.warning(str(exc))

    log.info("Pre-loading MTCNN…")
    get_mtcnn()   # warm up face detector so first request is fast

    app.run(host="0.0.0.0", port=5000, debug=False)
