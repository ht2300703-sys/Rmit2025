"""
Threshold Optimizer -- Member A (Huy)
Tim nguong quyet dinh toi uu hoa Macro-F1 tren OOF predictions.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


def search_best_threshold(
    y_true: np.ndarray,
    oof_probs: np.ndarray,
    step: float = 0.005,
    save_path: str = None,
) -> dict:
    """
    Quet toan bo khong gian nguong [0.05, 0.95] de tim T* toi uu hoa Macro-F1.

    Args:
        y_true:    Ground-truth binary labels (0/1).
        oof_probs: OOF probability scores (float in [0, 1]).
        step:      Step size for threshold grid search.
        save_path: Optional JSON path to save the result config.

    Returns:
        dict with keys: optimal_f1_threshold, optimal_f1_score, records_df
    """
    thresholds = np.arange(0.05, 0.95 + step, step)
    best_t = 0.5
    best_macro_f1 = -1.0
    records = []

    for t in thresholds:
        preds = (oof_probs >= t).astype(int)
        macro_f1 = f1_score(y_true, preds, average="macro")
        cm = confusion_matrix(y_true, preds)
        fn = cm[1, 0] if cm.shape == (2, 2) else 0

        records.append({
            "threshold": round(float(t), 4),
            "macro_f1": round(float(macro_f1), 4),
            "false_negatives": int(fn),
        })
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_t = float(t)

    df_res = pd.DataFrame(records)
    print(f"\n[Threshold Search] Nguong T* toi uu: {best_t:.4f} | Macro-F1 cao nhat: {best_macro_f1:.4f}")

    cfg = {
        "optimal_f1_threshold": best_t,
        "optimal_f1_score": round(best_macro_f1, 4),
    }
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
        print(f"[Threshold Search] Da luu ket qua vao: {save_path}")

    return cfg


def load_threshold(path: str, default: float = 0.5) -> float:
    """Tai nguong da luu tu file JSON. Fallback ve default neu file khong ton tai."""
    if path and __import__("os").path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return float(cfg.get("optimal_f1_threshold", default))
    return default
