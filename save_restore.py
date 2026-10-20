"""
Save & Restore utilities (per architecture diagram)
Luu model/checkpoint/submission len:
  1. Kaggle Dataset (push qua kaggle API)
  2. Hugging Face Hub (push_to_hub)
  3. Kaggle Notebook Save Version (huong dan)

Su dung: python save_restore.py [--mode hf|kaggle|check]
"""
import os
import sys
import json
import shutil
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Hugging Face Hub push
# ---------------------------------------------------------------------------

def push_to_hf_hub(
    local_model_dir: str,
    repo_id: str,
    private: bool = True,
    token: str = None,
) -> None:
    """
    Upload model checkpoint len Hugging Face Hub (private repo).
    Can: pip install huggingface_hub
    Bien moi truong: HF_TOKEN (hoac Kaggle Secret 'HF_TOKEN').
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("[HF Push] Chua cai huggingface_hub. Chay: pip install huggingface_hub")
        return

    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[HF Push] Thieu HF_TOKEN. Them vao Kaggle Secrets hoac bien moi truong.")
        return

    api = HfApi(token=hf_token)
    print(f"[HF Push] Tao/kiem tra repo: {repo_id} (private={private})...")
    try:
        create_repo(repo_id, private=private, token=hf_token, exist_ok=True)
    except Exception as e:
        print(f"[HF Push] create_repo warning: {e}")

    print(f"[HF Push] Dang upload {local_model_dir} -> {repo_id}...")
    api.upload_folder(
        folder_path=local_model_dir,
        repo_id=repo_id,
        token=hf_token,
    )
    print(f"[HF Push] Upload hoan tat! Model tai: https://huggingface.co/{repo_id}")


def pull_from_hf_hub(
    repo_id: str,
    local_dir: str = "./models/hf_downloaded",
    token: str = None,
) -> str:
    """
    Download model tu Hugging Face Hub.
    Tra ve duong dan thu muc local.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[HF Pull] Chua cai huggingface_hub. Chay: pip install huggingface_hub")
        return local_dir

    hf_token = token or os.environ.get("HF_TOKEN")
    print(f"[HF Pull] Downloading {repo_id} -> {local_dir}...")
    path = snapshot_download(repo_id=repo_id, local_dir=local_dir, token=hf_token)
    print(f"[HF Pull] Da tai ve: {path}")
    return path


# ---------------------------------------------------------------------------
# 2. Kaggle Dataset push
# ---------------------------------------------------------------------------

def push_to_kaggle_dataset(
    local_path: str,
    dataset_slug: str,
    title: str = "RMIT Hackathon 2025 Model Checkpoints",
    token_path: str = None,
) -> None:
    """
    Dong goi va push file/thu muc len Kaggle Dataset.
    Can: pip install kaggle
    Bien moi truong: KAGGLE_USERNAME, KAGGLE_KEY.

    Args:
        local_path:    File hoac thu muc can push.
        dataset_slug:  Ten dataset 'username/dataset-name'.
        title:         Tieu de hien thi tren Kaggle.
        token_path:    Duong dan den kaggle.json (mac dinh ~/.kaggle/kaggle.json).
    """
    try:
        import kaggle
    except ImportError:
        print("[Kaggle Push] Chua cai kaggle. Chay: pip install kaggle")
        return

    # Check credentials
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not username or not key:
        print("[Kaggle Push] Thieu KAGGLE_USERNAME hoac KAGGLE_KEY trong bien moi truong.")
        return

    # Create staging dir
    staging_dir = Path("./kaggle_staging")
    staging_dir.mkdir(exist_ok=True)

    src = Path(local_path)
    if src.is_file():
        shutil.copy2(src, staging_dir / src.name)
    elif src.is_dir():
        shutil.copytree(src, staging_dir / src.name, dirs_exist_ok=True)

    # Write dataset-metadata.json
    parts = dataset_slug.split("/")
    owner = parts[0] if len(parts) == 2 else username
    slug  = parts[-1].lower().replace(" ", "-").replace("_", "-")

    metadata = {
        "title": title,
        "id": f"{owner}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(staging_dir / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Kaggle Push] Pushing {local_path} -> {dataset_slug}...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
        api = KaggleApiExtended()
        api.authenticate()
        api.dataset_create_version(
            folder=str(staging_dir),
            version_notes="Auto push from train pipeline",
            quiet=False,
            convert_to_csv=False,
        )
        print(f"[Kaggle Push] Da push thanh cong: {dataset_slug}")
    except Exception as e:
        print(f"[Kaggle Push] Loi: {e}")


# ---------------------------------------------------------------------------
# 3. Checkpoint status checker
# ---------------------------------------------------------------------------

def check_artifacts() -> None:
    """Kiem tra trang thai cac file quan trong da luu chua."""
    important_files = [
        "./data/train.csv",
        "./data/test.csv",
        "./data/submission.csv",
        "./models/oof_mdeberta_v3.npy",
        "./models/oof_roberta.npy",
        "./models/threshold_ensemble.json",
        "challenge4.pkl",
    ]

    print("\n[Artifact Check] Trang thai cac file quan trong:")
    print("-" * 50)
    all_ok = True
    for fpath in important_files:
        exists = Path(fpath).exists()
        size_str = ""
        if exists:
            size = Path(fpath).stat().st_size
            size_str = f" ({size:,} bytes)" if size < 1_048_576 else f" ({size/1_048_576:.1f} MB)"
        status = "[OK]    " if exists else "[MISSING]"
        if not exists:
            all_ok = False
        print(f"  {status} {fpath}{size_str}")

    print("-" * 50)
    if all_ok:
        print("  Tat ca artifacts da san sang!")
    else:
        print("  Mot so file chua duoc tao. Hay chay train.py va predict.py truoc.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save/Restore utilities for RMIT Hackathon")
    parser.add_argument(
        "--mode",
        choices=["hf", "kaggle", "check", "hf_pull"],
        default="check",
    )
    parser.add_argument("--repo",   default="your-hf-username/rmit-hackathon-2025",
                        help="HuggingFace repo ID or Kaggle dataset slug")
    parser.add_argument("--local",  default="./models",
                        help="Local path for push/pull")
    parser.add_argument("--token",  default=None, help="API token override")
    args = parser.parse_args()

    if args.mode == "check":
        check_artifacts()
    elif args.mode == "hf":
        push_to_hf_hub(args.local, args.repo, private=True, token=args.token)
    elif args.mode == "hf_pull":
        pull_from_hf_hub(args.repo, args.local, token=args.token)
    elif args.mode == "kaggle":
        push_to_kaggle_dataset(args.local, args.repo)
