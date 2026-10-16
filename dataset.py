"""
Dataset module -- Member A (Huy)
PyTorch Dataset for prompt classification (Jailbreak vs Benign).
"""
import torch
from torch.utils.data import Dataset


class PromptDataset(Dataset):
    """
    Dataset cho bai toan phan loai prompt jailbreak.
    - texts: list of raw text strings
    - labels: list of int (0=benign, 1=jailbreak) or str labels; None for inference
    - tokenizer: HuggingFace tokenizer; None returns raw text dict
    - max_length: max token length for truncation
    """
    LABEL_MAP = {
        "benign": 0,
        "jailbreak": 1,
        "malicious": 1,
    }

    def __init__(self, texts, labels=None, tokenizer=None, max_length: int = 256):
        self.texts = [str(t) for t in texts]
        self.tokenizer = tokenizer
        self.max_length = max_length

        if labels is not None:
            processed_labels = []
            for lab in labels:
                if isinstance(lab, str):
                    processed_labels.append(self.LABEL_MAP.get(lab.lower(), 0))
                else:
                    processed_labels.append(int(lab))
            self.labels = processed_labels
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            item = {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
            }
            if "token_type_ids" in encoding:
                item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        else:
            item = {"text": text}

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item
