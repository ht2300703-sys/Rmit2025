# RMIT GenAI & Cybersecurity Hackathon 2025

**Team**: KAG_RMIT | **Notebook naming**: `SGS_<team>_C2`

## Thành viên & Phân công

| Thành viên | Vai trò | Module |
|:--|:--|:--|
| **Member A (Huy)** | Training mô hình phân loại chuỗi | `train.py`, `src/utils.py`, `src/dataset.py`, `src/threshold.py` |
| **Member B (Dũng)** | Chuẩn hóa ký tự, de-obfuscation, feature extraction | `src/preprocessing.py` |
| **Member C (Phan Khang)** | Red team, sinh dữ liệu đối kháng | `redteam_c4.py`, `C_PhanKhang_Delivery/` |
| **Member D (Phạm Khang)** | Prompt security, inference engine, guardrail | `src/prompt_security.py`, `src/inference.py`, `src/egress_guard.py`, `src/pipeline.py` |

## Chạy nhanh

```bash
# 1. Test pipeline structure (không cần GPU)
python train.py --model mdeberta --quick

# 2. Train đầy đủ
python train.py --model mdeberta --epochs 3 --folds 5

# 3. Tạo submission
python predict.py

# 4. Challenge 4 red team
python redteam_c4.py --model simulator

# 5. Kiểm tra artifacts
python save_restore.py --mode check
```

## Kiến trúc phòng thủ 4 tầng (Challenge 2 + D)

```
Input → [Ingress Guardrail] → [Prompt Hardening] → [GenAI Engine] → [Egress Guard] → Output
             ↓ block                                                      ↓ block
          (mDeBERTa)                                           (Canary + PII + Policy)
```

## Lưu trữ an toàn (chống mất khi Kaggle reset)

- **Kaggle Save Version**: Tự động commit toàn bộ `/kaggle/working/`
- **HF Hub**: `python save_restore.py --mode hf --repo user/rmit-2025`
- **Kaggle Dataset**: `python save_restore.py --mode kaggle --repo user/checkpoints`
