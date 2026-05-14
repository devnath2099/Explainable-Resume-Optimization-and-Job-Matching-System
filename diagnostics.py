import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import numpy as np
from dataset import prepare_data_splits
from utils_train import ID_TO_LABEL


def diagnose_dataset():
    splits = prepare_data_splits()
    for name in ["train", "val", "test"]:
        labels = splits[name]["labels"]
        dist = Counter(labels)
        print(f"\n[{name}] total={len(labels)}")
        for cls_id in sorted(dist):
            print(f"  {ID_TO_LABEL[cls_id]} ({cls_id}): {dist[cls_id]:>5}  "
                  f"({dist[cls_id]/len(labels)*100:5.1f}%)")


def diagnose_predictions(model, loader, device, max_samples=None):
    model.eval()
    all_logits, all_preds, all_labels, all_probs = [], [], [], []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_samples and i * loader.batch_size >= max_samples:
                break
            resume_ids = batch["resume_input_ids"].to(device)
            resume_mask = batch["resume_mask"].to(device)
            job_ids = batch["job_input_ids"].to(device)
            job_mask = batch["job_mask"].to(device)
            labels = batch["label"]

            outputs = model(resume_ids, resume_mask, job_ids, job_mask)
            logits = outputs["logits"]
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_logits.append(logits.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()

    # --- Confusion matrix (absolute) ---
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(ID_TO_LABEL.values()),
                yticklabels=list(ID_TO_LABEL.values()))
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.show()

    # --- Normalized confusion matrix ---
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=list(ID_TO_LABEL.values()),
                yticklabels=list(ID_TO_LABEL.values()))
    plt.title("Confusion Matrix (row-normalized)")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.show()

    # --- Per-class metrics ---
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=list(ID_TO_LABEL.values())))

    # --- Prediction distribution ---
    pred_dist = Counter(preds.tolist())
    label_dist = Counter(labels.tolist())
    print("\nLabel distribution in this batch:")
    for c in sorted(label_dist):
        print(f"  True {ID_TO_LABEL[c]}: {label_dist[c]}")
    print("Prediction distribution:")
    for c in sorted(pred_dist):
        print(f"  Pred {ID_TO_LABEL[c]}: {pred_dist[c]}")

    # --- Confidence analysis ---
    max_probs = probs.max(axis=1)
    print(f"\nMean confidence: {max_probs.mean():.4f}")
    print(f"Confidence < 0.6: {(max_probs < 0.6).mean()*100:.1f}%")
    for c in range(3):
        mask = labels == c
        if mask.sum() > 0:
            print(f"  {ID_TO_LABEL[c]} — mean conf: {max_probs[mask].mean():.4f}")

    return cm, probs


# --- Per-class confusion analysis ---
def confusion_by_class(cm, labels):
    """Print which classes each true label gets confused with."""
    for true_idx in range(cm.shape[0]):
        row = cm[true_idx]
        total = row.sum()
        if total == 0:
            continue
        print(f"\nTrue = {ID_TO_LABEL[true_idx]} (n={total}):")
        for pred_idx in range(len(row)):
            pct = row[pred_idx] / total * 100
            print(f"  → predicted {ID_TO_LABEL[pred_idx]}: {row[pred_idx]} ({pct:.1f}%)")
