"""
Challenge 2 -- Ingress Classifier Training (Member A: Huy)
Huan luyen mo hinh phan loai prompt Jailbreak vs Benign voi Stratified K-Fold.

Su dung: python train.py [--quick] [--model mdeberta|roberta] [--epochs N]
"""
import os
import gc
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_cosine_schedule_with_warmup,
)
from tqdm.auto import tqdm

from src.utils import seed_everything, compute_metrics, detailed_metrics
from src.dataset import PromptDataset
from src.threshold import search_best_threshold
from src.preprocessing import clean_and_deobfuscate_text


# ---------------------------------------------------------------------------
# Paths (tuong thich Kaggle va may cuc bo)
# ---------------------------------------------------------------------------
IS_KAGGLE = os.path.exists("/kaggle/working")
BASE_DATA = "/kaggle/input" if IS_KAGGLE else "./data"
BASE_WORK = "/kaggle/working/hackathon-security" if IS_KAGGLE else "."
TRAIN_FILE = (
    "./train.csv" if os.path.exists("./train.csv")
    else "./data/train.csv" if os.path.exists("./data/train.csv")
    else os.path.join(BASE_DATA, "train.csv")
)
TEST_FILE = (
    "./test.csv" if os.path.exists("./test.csv")
    else "./data/test.csv" if os.path.exists("./data/test.csv")
    else os.path.join(BASE_DATA, "test.csv")
)


MODEL_REGISTRY = {
    "mdeberta": "microsoft/mdeberta-v3-base",
    "roberta":  "roberta-base",
    "distilbert": "distilbert-base-uncased",
}


def run_training(
    model_key: str = "mdeberta",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    max_length: int = 256,
    n_folds: int = 5,
    quick_run: bool = False,
    save_hf_model: bool = False,
):
    """
    Pipeline huan luyen chinh:
    1. Doc train.csv, lam sach van ban, tao fold.
    2. Loop qua n_folds voi Stratified K-Fold.
    3. Luu OOF probs, test probs, checkpoint tung fold.
    4. Tim nguong toi uu T* tren OOF.

    Args:
        model_key:      Key trong MODEL_REGISTRY ('mdeberta', 'roberta', ...).
        epochs:         So epoch moi fold.
        batch_size:     Batch size huan luyen.
        lr:             Learning rate ban dau.
        max_length:     Max token length.
        n_folds:        So fold Stratified K-Fold.
        quick_run:      True = chi chay 1 fold voi 32 mau (kiem tra pipeline).
        save_hf_model:  True = luu toan bo HuggingFace model (lon hon, de push HF Hub).
    """
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = MODEL_REGISTRY.get(model_key, model_key)
    model_alias = model_key

    print(f"\n{'='*60}")
    print(f"RMIT Hackathon 2025 -- Challenge 2: Ingress Classifier")
    print(f"Model : {model_name}")
    print(f"Device: {device}")
    print(f"Folds : {'1 (quick_run)' if quick_run else n_folds}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # 1. Load & preprocess data
    # ------------------------------------------------------------------ #
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(
            f"Khong tim thay {TRAIN_FILE}.\n"
            "Hay dat train.csv vao ./data/ hoac ket noi Kaggle dataset."
        )

    df_train = pd.read_csv(TRAIN_FILE)
    print(f"[Data] Train samples: {len(df_train)}")

    # De-obfuscate text
    df_train["text"] = df_train["text"].apply(clean_and_deobfuscate_text)

    label_map = {"benign": 0, "jailbreak": 1}
    df_train["label_num"] = df_train["label"].map(
        lambda l: label_map.get(str(l).lower(), 0)
    )
    print(f"[Data] Class distribution:\n{df_train['label_num'].value_counts()}\n")

    # Load test if available
    df_test = None
    if os.path.exists(TEST_FILE):
        df_test = pd.read_csv(TEST_FILE)
        df_test["text"] = df_test["text"].apply(clean_and_deobfuscate_text)
        print(f"[Data] Test samples: {len(df_test)}")

    # ------------------------------------------------------------------ #
    # 2. Create folds
    # ------------------------------------------------------------------ #
    fold_file = "./data/train_folds.csv"
    if "fold" not in df_train.columns:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        df_train["fold"] = -1
        for fold, (_, val_idx) in enumerate(skf.split(df_train, df_train["label_num"])):
            df_train.loc[val_idx, "fold"] = fold
        os.makedirs("./data", exist_ok=True)
        df_train.to_csv(fold_file, index=False)
        print(f"[Folds] Da luu train_folds.csv")

    # ------------------------------------------------------------------ #
    # 3. Tokenizer & directories
    # ------------------------------------------------------------------ #
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    save_dir = f"./models/{model_alias}"
    os.makedirs(save_dir, exist_ok=True)

    oof_probs  = np.zeros(len(df_train))
    test_probs = np.zeros(len(df_test)) if df_test is not None else None

    test_loader = None
    if df_test is not None:
        test_ds = PromptDataset(df_test["text"].values, tokenizer=tokenizer, max_length=max_length)
        test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    num_folds = 1 if quick_run else n_folds

    # ------------------------------------------------------------------ #
    # 4. K-Fold training loop
    # ------------------------------------------------------------------ #
    for fold in range(num_folds):
        print(f"\n{'─'*50}")
        print(f"FOLD {fold + 1} / {num_folds}")
        print(f"{'─'*50}")

        train_df = df_train[df_train["fold"] != fold].reset_index(drop=True)
        val_df   = df_train[df_train["fold"] == fold].reset_index(drop=True)
        val_idx  = df_train[df_train["fold"] == fold].index.values

        if quick_run:
            train_df = train_df.head(32)
            val_df   = val_df.head(16)
            val_idx  = val_idx[:16]

        train_loader = DataLoader(
            PromptDataset(train_df["text"].values, train_df["label_num"].values,
                          tokenizer, max_length),
            batch_size=batch_size, shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            PromptDataset(val_df["text"].values, val_df["label_num"].values,
                          tokenizer, max_length),
            batch_size=batch_size * 2, shuffle=False, num_workers=0,
        )

        save_fold_path = f"{save_dir}/{model_alias}_fold{fold}.pt"
        config = AutoConfig.from_pretrained(model_name, num_labels=2)
        model  = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config
        ).to(device)

        if os.path.exists(save_fold_path):
            print(f"[Resume] Phat hien checkpoint {save_fold_path} da hoan thanh. Dang tinh toan OOF va bo qua train fold nay...")
            model.load_state_dict(torch.load(save_fold_path, map_location=device))
            model.eval()
            val_preds = []
            with torch.no_grad():
                for batch in val_loader:
                    logits = model(
                        batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                    ).logits
                    val_preds.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            val_preds = np.array(val_preds)
            oof_probs[val_idx] = val_preds
            metrics = detailed_metrics(val_df["label_num"].values, val_preds)
            print(f"[Resume] Fold {fold + 1} da load thanh cong: Macro-F1 = {metrics['macro_f1']:.4f} | AUC = {metrics['auc']:.4f}")

            if test_loader is not None and test_probs is not None:
                test_fold_preds = []
                with torch.no_grad():
                    for batch in test_loader:
                        logits = model(
                            batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device),
                        ).logits
                        test_fold_preds.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
                test_probs += np.array(test_fold_preds) / num_folds
            del model
            torch.cuda.empty_cache()
            gc.collect()
            continue

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * (1 if quick_run else epochs)
        scheduler  = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )
        criterion = torch.nn.CrossEntropyLoss()

        best_val_auc = 0.0
        for epoch in range(1 if quick_run else epochs):
            # -- Train
            model.train()
            train_loss = 0.0
            for batch in tqdm(train_loader, desc=f"  Epoch {epoch+1} Train", leave=False):
                optimizer.zero_grad()
                out  = model(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                loss = criterion(out.logits, batch["label"].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

            # -- Val
            model.eval()
            val_preds = []
            with torch.no_grad():
                for batch in val_loader:
                    logits = model(
                        batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                    ).logits
                    val_preds.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())

            val_preds = np.array(val_preds)
            metrics   = detailed_metrics(val_df["label_num"].values, val_preds)
            avg_loss  = train_loss / len(train_loader)

            print(
                f"  Epoch {epoch+1} | Loss: {avg_loss:.4f} "
                f"| Macro-F1: {metrics['macro_f1']:.4f} "
                f"| AUC: {metrics['auc']:.4f}"
            )

            if metrics["auc"] >= best_val_auc:
                best_val_auc = metrics["auc"]
                torch.save(
                    model.state_dict(),
                    f"{save_dir}/{model_alias}_fold{fold}.pt",
                )

        oof_probs[val_idx] = val_preds

        # -- Test inference (last epoch)
        if test_loader is not None and test_probs is not None:
            model.eval()
            test_fold_preds = []
            with torch.no_grad():
                for batch in test_loader:
                    logits = model(
                        batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                    ).logits
                    test_fold_preds.extend(
                        torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                    )
            test_probs += np.array(test_fold_preds) / num_folds

        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------ #
    # 5. Luu OOF, test probs va tim nguong toi uu
    # ------------------------------------------------------------------ #
    np.save(f"./models/oof_{model_alias}.npy", oof_probs)
    print(f"\n[Save] OOF probs -> ./models/oof_{model_alias}.npy")

    if test_probs is not None:
        np.save(f"./models/test_{model_alias}.npy", test_probs)
        print(f"[Save] Test probs -> ./models/test_{model_alias}.npy")

    cfg = search_best_threshold(
        df_train["label_num"].values[:len(oof_probs)],
        oof_probs,
        save_path=f"./models/threshold_{model_alias}.json",
    )
    print(f"\n[Done] Nguong toi uu T* = {cfg['optimal_f1_threshold']:.4f} | Macro-F1 = {cfg['optimal_f1_score']:.4f}")
    print(f"[Done] Mo hinh da luu tai: ./models/{model_alias}/")
    return cfg


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RMIT Hackathon 2025 -- C2 Training")
    parser.add_argument("--model",   default="mdeberta", choices=list(MODEL_REGISTRY.keys()),
                        help="Model key: mdeberta | roberta | distilbert")
    parser.add_argument("--epochs",  type=int, default=3)
    parser.add_argument("--batch",   type=int, default=8)
    parser.add_argument("--lr",      type=float, default=2e-5)
    parser.add_argument("--folds",   type=int, default=5)
    parser.add_argument("--maxlen",  type=int, default=256)
    parser.add_argument("--quick",   action="store_true",
                        help="Quick run: 1 fold, 32 samples (pipeline test)")
    args = parser.parse_args()

    run_training(
        model_key=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        max_length=args.maxlen,
        n_folds=args.folds,
        quick_run=args.quick,
    )
