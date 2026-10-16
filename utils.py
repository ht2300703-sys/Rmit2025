"""
Utility module -- Member A (Huy)
Seed, metrics, and helper functions for model training.
"""
import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix


def seed_everything(seed: int = 42) -> None:
    """Fix tat ca random seeds de ket qua co the tai tao."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(eval_pred) -> dict:
    """
    Tinh Macro-F1 va AUC tu logits va labels.
    eval_pred: tuple (logits, labels) tuong thich voi HuggingFace Trainer.
    """
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    probs = torch.softmax(torch.tensor(predictions), dim=-1)[:, 1].numpy()
    preds = np.argmax(predictions, axis=1)

    macro_f1 = f1_score(labels, preds, average="macro")
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.5
    return {"macro_f1": macro_f1, "auc": auc}


def detailed_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Bao cao day du cac chi so phan loai.
    Returns: macro_f1, auc, precision, recall, confusion_matrix.
    """
    y_pred = (y_pred_probs >= threshold).astype(int)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except Exception:
        auc = 0.5
    return {
        "macro_f1": round(macro_f1, 4),
        "auc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "confusion_matrix": cm.tolist(),
        "threshold": threshold,
    }
