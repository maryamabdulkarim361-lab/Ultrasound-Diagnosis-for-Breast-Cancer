"""
Simple Flask web interface for the BUSI ultrasound classifier.

Usage:
    python3 app.py
Then open http://127.0.0.1:5000 in your browser.

Requires a trained model file (default: best_model.pth) in this same folder.
Train one first with train.py if you haven't already.

If best_model.pth was trained with a non-default architecture, set MODEL_ARCH
before starting:
    MODEL_ARCH=resnet50 python3 app.py

If the 'grad-cam' package is installed, a heatmap overlay is shown alongside
each prediction automatically; if not, that panel is simply omitted.
"""

import os
import uuid

from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import torch.nn as nn
import numpy as np
from PIL import Image

from dataset import CLASS_NAMES
from report_generator import RECOMMENDATION, predict_with_sensitivity
from gradcam_utils import generate_gradcam_overlay, GRADCAM_AVAILABLE

APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_DIR, "static", "uploads")
MODEL_PATH = os.path.join(APP_DIR, "best_model.pth")
# Must match the --arch used when training best_model.pth. Override with:
#   MODEL_ARCH=resnet50 python3 app.py
MODEL_ARCH = os.environ.get("MODEL_ARCH", "resnet18")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None  # lazy-loaded so the app still starts even if the model file is missing yet


def build_model(arch="resnet18", num_classes=3):
    arch_map = {
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
        "efficientnet_b2": models.efficientnet_b2,
    }
    if arch not in arch_map:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {list(arch_map)}.")
    model = arch_map[arch](weights=None)
    if arch in ("resnet18", "resnet50"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:  # efficientnet_b2
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at '{MODEL_PATH}'. Run train.py first."
            )
        m = build_model(arch=MODEL_ARCH, num_classes=len(CLASS_NAMES)).to(device)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        m.eval()
        _model = m
    return _model


def preprocess(image_path):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    rgb_0_1 = np.array(image).astype(np.float32) / 255.0
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    tensor = transforms.Compose([transforms.ToTensor(), normalize])(image).unsqueeze(0)
    return tensor, rgb_0_1


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, error=None, gradcam_available=GRADCAM_AVAILABLE)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("scan")
    if not file or file.filename == "":
        return render_template("index.html", result=None,
                                error="Please choose an ultrasound image file first.",
                                gradcam_available=GRADCAM_AVAILABLE)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg"):
        return render_template("index.html", result=None,
                                error="Please upload a .png, .jpg, or .jpeg image.",
                                gradcam_available=GRADCAM_AVAILABLE)

    saved_name = f"{uuid.uuid4().hex}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)
    file.save(saved_path)

    try:
        model = get_model()
    except FileNotFoundError as e:
        return render_template("index.html", result=None, error=str(e),
                                gradcam_available=GRADCAM_AVAILABLE)

    image_tensor, rgb_0_1 = preprocess(saved_path)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)[0]

    probs_list = probs.cpu().tolist()
    # Use standard argmax for the main web-interface prediction (dynamic
    # thresholding is available via predict.py for command-line sensitivity tuning)
    predicted_idx = int(torch.argmax(probs).item())
    confidence_pct = round(probs_list[predicted_idx] * 100, 1)
    predicted_class = CLASS_NAMES[predicted_idx]

    class_probs = [
        {"name": CLASS_NAMES[i], "pct": round(probs_list[i] * 100, 1)}
        for i in range(len(CLASS_NAMES))
    ]
    class_probs.sort(key=lambda x: -x["pct"])

    gradcam_url = None
    if GRADCAM_AVAILABLE:
        overlay = generate_gradcam_overlay(
            model, MODEL_ARCH, image_tensor, rgb_0_1, target_category=predicted_idx
        )
        if overlay is not None:
            gradcam_name = f"gradcam_{saved_name}"
            Image.fromarray(overlay).save(os.path.join(UPLOAD_DIR, gradcam_name))
            gradcam_url = f"/static/uploads/{gradcam_name}"

    result = {
        "image_url": f"/static/uploads/{saved_name}",
        "gradcam_url": gradcam_url,
        "predicted_class": predicted_class,
        "confidence_pct": confidence_pct,
        "class_probs": class_probs,
        "recommendation": RECOMMENDATION[predicted_class],
    }
    return render_template("index.html", result=result, error=None,
                            gradcam_available=GRADCAM_AVAILABLE)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
