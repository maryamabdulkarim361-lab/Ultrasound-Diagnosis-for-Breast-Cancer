"""
Load a trained model and generate a report for a single ultrasound image.

Basic usage:
    python predict.py --image path/to/scan.png --model best_model.pth

If the model was trained with a non-default architecture (--arch resnet50 or
--arch efficientnet_b2 in train.py), pass the same --arch here so the model
shapes match.

Optional: lower the malignant sensitivity threshold (trades some precision
for higher malignant recall):
    python predict.py --image scan.png --model best_model.pth --malignant_threshold 0.35

Optional: save a Grad-CAM heatmap showing which region of the image most
influenced the prediction (requires 'pip install grad-cam'):
    python predict.py --image scan.png --model best_model.pth --gradcam
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

from dataset import CLASS_NAMES
from report_generator import generate_report, predict_with_sensitivity
from gradcam_utils import generate_gradcam_overlay, GRADCAM_AVAILABLE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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


def load_image(image_path):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    rgb_0_1 = np.array(image).astype(np.float32) / 255.0  # for Grad-CAM overlay background

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    tensor = transforms.Compose([transforms.ToTensor(), normalize])(image).unsqueeze(0)
    return tensor, rgb_0_1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to ultrasound image")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .pth file")
    parser.add_argument("--patient_id", type=str, default=None)
    parser.add_argument("--scan_id", type=str, default=None)
    parser.add_argument("--arch", type=str, default="resnet18",
                         choices=["resnet18", "resnet50", "efficientnet_b2"],
                         help="Must match the architecture used in train.py for this model file")
    parser.add_argument("--malignant_threshold", type=float, default=None,
                         help="Optional: flag malignant if its probability alone clears this "
                              "threshold (e.g. 0.35), instead of picking the single highest "
                              "class. Trades precision for recall on malignant cases. Omit for "
                              "standard argmax behavior.")
    parser.add_argument("--gradcam", action="store_true",
                         help="Save a Grad-CAM heatmap overlay (requires 'pip install grad-cam')")
    parser.add_argument("--gradcam_output", type=str, default="gradcam_overlay.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(arch=args.arch, num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    image_tensor, rgb_0_1 = load_image(args.image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)[0]

    probs_list = probs.cpu().tolist()

    if args.malignant_threshold is not None:
        predicted_idx, malignant_prob = predict_with_sensitivity(
            probs_list, malignant_threshold=args.malignant_threshold
        )
        confidence = probs_list[predicted_idx]
        threshold_used = args.malignant_threshold
    else:
        predicted_idx = int(torch.argmax(probs).item())
        confidence = probs_list[predicted_idx]
        threshold_used = None

    report = generate_report(
        predicted_class_idx=predicted_idx,
        confidence=confidence,
        patient_id=args.patient_id,
        scan_id=args.scan_id,
        malignant_threshold=threshold_used,
    )
    print(report)
    print(f"\nFull probabilities: "
          f"{dict(zip(CLASS_NAMES, [round(p, 4) for p in probs_list]))}")

    if args.gradcam:
        if not GRADCAM_AVAILABLE:
            print("\n[Grad-CAM] 'grad-cam' package not installed -- skipping heatmap. "
                  "Install with: pip install grad-cam")
        else:
            overlay = generate_gradcam_overlay(
                model, args.arch, image_tensor, rgb_0_1, target_category=predicted_idx
            )
            if overlay is not None:
                Image.fromarray(overlay).save(args.gradcam_output)
                print(f"\n[Grad-CAM] Saved heatmap overlay to {args.gradcam_output}")


if __name__ == "__main__":
    main()
