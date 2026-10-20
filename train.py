"""
Train a breast-ultrasound classifier on the BUSI dataset.

RECOMMENDED usage (ResNet-50 backbone, offline pretrained weights, Focal Loss,
Albumentations augmentation -- all headline improvements applied):
    python train.py --data_dir data/Dataset_BUSI_with_GT --epochs 35 \
        --arch resnet50 --pretrained_path resnet50-imagenet.pth

If you don't have an offline weights file, but this machine CAN reach
download.pytorch.org, drop --pretrained_path and it will download automatically:
    python train.py --data_dir data/Dataset_BUSI_with_GT --epochs 35 --arch resnet50

If neither is possible, train from scratch (lower accuracy, but works):
    python train.py --data_dir data/Dataset_BUSI_with_GT --epochs 35 --no_pretrained

Architecture options (--arch): resnet18 (fastest on CPU, default), resnet50,
efficientnet_b2 (best accuracy potential, slowest on CPU).

This script:
  - Uses Focal Loss (gamma=2.0 default) with square-root-smoothed class
    weights by default -- gentler on imbalance than raw inverse-frequency
    weighting, which was found to overcorrect in an earlier run (malignant
    recall dropped when normal was up-weighted too aggressively).
  - Applies Albumentations augmentation (CLAHE, random rotation, grid
    distortion) if the 'albumentations' package is installed; falls back to
    a torchvision pipeline otherwise.
  - Reports Malignant Recall (Sensitivity) and AUC-ROC as headline metrics,
    since a missed malignant case (false negative) is the costliest error in
    a screening context -- more important than overall accuracy.
  - Saves an AUC-ROC curve plot (roc_curve.png) after final evaluation.
  - Defaults num_workers=0 for DataLoaders, which is the safest setting on
    macOS/CPU-only machines (higher values can cause multiprocessing crashes).
"""

import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)

from dataset import BUSIDataset, CLASS_NAMES

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

MALIGNANT_IDX = CLASS_NAMES.index("malignant")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017). Down-weights easy/well-classified examples
    and focuses training on hard, confusing ones -- handles class imbalance
    more gently than raw class-weighted Cross-Entropy.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        ce = F.nll_loss(log_probs, targets, weight=self.alpha, reduction="none")
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_term = (1 - pt) ** self.gamma
        return (focal_term * ce).mean()


def build_model(arch="resnet18", num_classes=3, pretrained=True, pretrained_path=None):
    arch_map = {
        "resnet18": (models.resnet18, "ResNet18_Weights"),
        "resnet50": (models.resnet50, "ResNet50_Weights"),
        "efficientnet_b2": (models.efficientnet_b2, "EfficientNet_B2_Weights"),
    }
    if arch not in arch_map:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {list(arch_map)}.")
    base_fn, weights_enum_name = arch_map[arch]

    if pretrained_path:
        model = base_fn(weights=None)
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded offline pretrained weights from '{pretrained_path}' for {arch}")
    elif pretrained:
        try:
            weights_enum = getattr(models, weights_enum_name).IMAGENET1K_V1
            model = base_fn(weights=weights_enum)
            print(f"Downloaded ImageNet-pretrained weights for {arch}")
        except Exception as e:
            print(f"Could not download pretrained weights for {arch} ({e}). "
                  f"Falling back to training from scratch.")
            model = base_fn(weights=None)
    else:
        model = base_fn(weights=None)

    if arch in ("resnet18", "resnet50"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "efficientnet_b2":
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def get_torchvision_transforms():
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def get_albumentations_transforms():
    train_tf = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    eval_tf = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    return train_tf, eval_tf


def get_dataset_class(use_albumentations):
    if use_albumentations:
        from dataset import BUSIDatasetAlbumentations
        return BUSIDatasetAlbumentations
    return BUSIDataset


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))
    report = classification_report(
        all_labels, all_preds, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, zero_division=0
    )
    malignant_recall = recall[MALIGNANT_IDX]

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", labels=list(range(len(CLASS_NAMES))))
    except ValueError:
        auc = float("nan")

    return {
        "acc": acc, "precision": precision, "recall": recall, "f1": f1,
        "cm": cm, "report": report, "malignant_recall": malignant_recall,
        "auc": auc, "labels": all_labels, "probs": all_probs,
    }


def save_roc_curve(labels, probs, output_path="roc_curve.png"):
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available -- skipping ROC curve plot.")
        return

    labels = np.array(labels)
    probs = np.array(probs)

    plt.figure(figsize=(6, 6))
    for idx, class_name in enumerate(CLASS_NAMES):
        binary_labels = (labels == idx).astype(int)
        if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
            continue
        fpr, tpr, _ = roc_curve(binary_labels, probs[:, idx])
        auc_val = roc_auc_score(binary_labels, probs[:, idx])
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc_val:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve \u2014 One-vs-Rest (Test Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved ROC curve plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to Dataset_BUSI_with_GT folder")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="best_model.pth")
    parser.add_argument("--arch", type=str, default="resnet18",
                         choices=["resnet18", "resnet50", "efficientnet_b2"],
                         help="Backbone architecture. resnet18 is fastest on CPU; "
                              "resnet50/efficientnet_b2 may improve accuracy but "
                              "train noticeably slower on CPU-only machines.")
    parser.add_argument("--no_pretrained", action="store_true",
                         help="Train from scratch instead of using ImageNet weights")
    parser.add_argument("--pretrained_path", type=str, default=None,
                         help="Path to a locally downloaded ImageNet weights .pth file")
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "ce"])
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=0,
                         help="DataLoader worker processes. 0 (default) is safest on "
                              "macOS/CPU-only machines.")
    parser.add_argument("--no_albumentations", action="store_true",
                         help="Use the basic torchvision augmentation pipeline instead of "
                              "Albumentations (CLAHE / grid distortion).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)}) | architecture: {args.arch}")
    else:
        print(f"Using device: {device} | architecture: {args.arch}")

    use_albumentations = ALBUMENTATIONS_AVAILABLE and not args.no_albumentations
    if use_albumentations:
        train_tf, eval_tf = get_albumentations_transforms()
        print("Using Albumentations augmentation pipeline (CLAHE, rotation, grid distortion).")
    else:
        train_tf, eval_tf = get_torchvision_transforms()
        if not ALBUMENTATIONS_AVAILABLE:
            print("albumentations not installed -- using torchvision augmentation pipeline. "
                  "Run 'pip install albumentations' to enable CLAHE/grid-distortion augmentation.")
        else:
            print("Using torchvision augmentation pipeline (--no_albumentations was set).")

    DatasetClass = get_dataset_class(use_albumentations)

    full_dataset = DatasetClass(args.data_dir, transform=eval_tf)
    n_total = len(full_dataset)
    n_val = int(0.15 * n_total)
    n_test = int(0.15 * n_total)
    n_train = n_total - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    train_ds.dataset = DatasetClass(args.data_dir, transform=train_tf)

    print(f"Total images: {n_total} | train: {n_train} | val: {n_val} | test: {n_test}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    model = build_model(
        arch=args.arch,
        num_classes=len(CLASS_NAMES),
        pretrained=not args.no_pretrained,
        pretrained_path=args.pretrained_path,
    ).to(device)

    weights_tensor = None
    if not args.no_class_weights:
        train_labels = [full_dataset.samples[i][1] for i in train_ds.indices]
        class_counts = np.array([train_labels.count(i) for i in range(len(CLASS_NAMES))])
        raw_weights = class_counts.sum() / (len(CLASS_NAMES) * class_counts)
        class_weights = np.sqrt(raw_weights)
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        print(f"Class counts (train): {dict(zip(CLASS_NAMES, class_counts))}")
        print(f"Class weights applied (sqrt-smoothed): "
              f"{dict(zip(CLASS_NAMES, class_weights.round(2)))}")

    if args.loss == "focal":
        criterion = FocalLoss(alpha=weights_tensor, gamma=args.focal_gamma)
        print(f"Using Focal Loss (gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        print("Using Cross-Entropy Loss")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.patience
    )

    best_score = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / n_train
        val_metrics = evaluate(model, val_loader, device)
        val_acc = val_metrics["acc"]
        val_malignant_recall = val_metrics["malignant_recall"]

        score = 0.5 * val_acc + 0.5 * val_malignant_recall
        scheduler.step(score)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{args.epochs} | train_loss: {train_loss:.4f} | "
              f"val_acc: {val_acc:.4f} | val_malignant_recall: {val_malignant_recall:.4f} | "
              f"lr: {current_lr:.2e}")

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())

    print(f"\nBest combined validation score (0.5*acc + 0.5*malignant_recall): {best_score:.4f}")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.output)
    print(f"Saved best model to {args.output}")

    print("\n=== Final Test Set Evaluation ===")
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_metrics['acc']:.4f}")
    print(f"Malignant recall / Sensitivity (headline safety metric): "
          f"{test_metrics['malignant_recall']:.4f}")
    print(f"AUC-ROC (macro, one-vs-rest): {test_metrics['auc']:.4f}\n")
    print(test_metrics["report"])
    print("Confusion matrix (rows=true, cols=predicted):")
    print(f"Classes order: {CLASS_NAMES}")
    print(test_metrics["cm"])

    if test_metrics["malignant_recall"] < 0.90:
        print(f"\n[Note] Malignant recall ({test_metrics['malignant_recall']:.1%}) is below the "
              f"90-95% target commonly used for cancer screening sensitivity. Consider more "
              f"epochs, offline pretrained weights, or a larger architecture (--arch resnet50).")

    save_roc_curve(test_metrics["labels"], test_metrics["probs"], output_path="roc_curve.png")


if __name__ == "__main__":
    main()
