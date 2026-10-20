"""
OPTIONAL: Stratified K-Fold Cross-Validation for the BUSI classifier.

This is a SEPARATE script from train.py. It doesn't produce the model used
by predict.py/app.py -- it exists to give a more statistically reliable
estimate of how the model performs on this small (~780 image) dataset than
a single train/val/test split can, which is valuable evidence for a project
report.

IMPORTANT -- TIME COST: this trains the model K times from scratch (default
K=5). On a CPU-only machine, this takes roughly K times as long as a single
run of train.py. With --epochs 15 (recommended for k-fold, not 35), expect
very roughly 1.5-3 hours total on a laptop CPU. Run this when you have time
to leave it running (e.g. overnight), not right before a deadline.

Usage:
    python train_kfold.py --data_dir data/Dataset_BUSI_with_GT --folds 5 --epochs 15
"""

import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score
)

from dataset import BUSIDataset, CLASS_NAMES
from train import FocalLoss, build_model, get_torchvision_transforms

MALIGNANT_IDX = CLASS_NAMES.index("malignant")


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
    _, recall, _, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", labels=list(range(len(CLASS_NAMES))))
    except ValueError:
        auc = float("nan")
    return acc, recall[MALIGNANT_IDX], auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15,
                         help="Kept lower than train.py's default since this multiplies "
                              "by --folds. Increase only if you have time to spare.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--arch", type=str, default="resnet18",
                         choices=["resnet18", "resnet50", "efficientnet_b2"])
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--pretrained_path", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | {args.folds}-fold stratified cross-validation | "
          f"arch: {args.arch} | epochs/fold: {args.epochs}")
    print("This will take a while -- see the time-cost note in this script's docstring.\n")

    train_tf, eval_tf = get_torchvision_transforms()
    full_dataset_eval = BUSIDataset(args.data_dir, transform=eval_tf)
    full_dataset_train = BUSIDataset(args.data_dir, transform=train_tf)
    targets = [label for _, label in full_dataset_eval.samples]

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    fold_accs, fold_recalls, fold_aucs = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets), start=1):
        print(f"\n========== Fold {fold}/{args.folds} ==========")

        train_sub = Subset(full_dataset_train, train_idx)
        val_sub = Subset(full_dataset_eval, val_idx)
        train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = build_model(
            arch=args.arch, num_classes=len(CLASS_NAMES),
            pretrained=not args.no_pretrained, pretrained_path=args.pretrained_path,
        ).to(device)

        fold_train_labels = [targets[i] for i in train_idx]
        class_counts = np.array([fold_train_labels.count(i) for i in range(len(CLASS_NAMES))])
        raw_weights = class_counts.sum() / (len(CLASS_NAMES) * class_counts)
        weights_tensor = torch.tensor(np.sqrt(raw_weights), dtype=torch.float32).to(device)

        criterion = FocalLoss(alpha=weights_tensor, gamma=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            if epoch == args.epochs or epoch % 5 == 0:
                print(f"  epoch {epoch}/{args.epochs} | train_loss: {running_loss/len(train_idx):.4f}")

        acc, malignant_recall, auc = evaluate(model, val_loader, device)
        print(f"Fold {fold} result -> accuracy: {acc:.4f} | "
              f"malignant_recall: {malignant_recall:.4f} | AUC-ROC: {auc:.4f}")

        fold_accs.append(acc)
        fold_recalls.append(malignant_recall)
        fold_aucs.append(auc)

    print("\n========== Cross-Validation Summary ==========")
    print(f"Accuracy:         {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")
    print(f"Malignant recall: {np.mean(fold_recalls):.4f} (+/- {np.std(fold_recalls):.4f})")
    print(f"AUC-ROC:          {np.nanmean(fold_aucs):.4f} (+/- {np.nanstd(fold_aucs):.4f})")
    print("\nThese averaged, cross-validated numbers are more robust evidence of real "
          "performance than a single train/val/test split, and are worth reporting "
          "alongside (not instead of) the main train.py results.")


if __name__ == "__main__":
    main()
