import os
import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    batch_size: int = 4
    max_seq_length: int = 128
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 0
    seed: int = 42
    save_dir: str = "saved_models"
    log_interval: int = 10
    model_name: str = "distilbert-base-uncased"


# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------
ID_TO_LABEL: Dict[int, str] = {0: "No Fit", 1: "Potential Fit", 2: "Good Fit"}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Dataset: tokenizes resume and job description separately
# ---------------------------------------------------------------------------
class SeparateTokenizationDataset(Dataset):
    def __init__(
        self,
        resume_texts: List[str],
        job_texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128,
    ):
        self.resume_texts = resume_texts
        self.job_texts = job_texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        resume = str(self.resume_texts[idx])
        job = str(self.job_texts[idx])
        label = int(self.labels[idx])

        resume_enc = self.tokenizer(
            resume,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        job_enc = self.tokenizer(
            job,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "resume_input_ids": resume_enc["input_ids"].squeeze(0),
            "resume_mask": resume_enc["attention_mask"].squeeze(0),
            "job_input_ids": job_enc["input_ids"].squeeze(0),
            "job_mask": job_enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


if __name__ == "__main__":
    seed_everything(42)
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"ID_TO_LABEL: {ID_TO_LABEL}")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    ds = SeparateTokenizationDataset(
        resume_texts=["Experienced Python developer."],
        job_texts=["Looking for a Python engineer."],
        labels=[1],
        tokenizer=tokenizer,
        max_length=16,
    )
    item = ds[0]
    print(f"resume_input_ids: {item['resume_input_ids'].shape}")
    print(f"job_input_ids:    {item['job_input_ids'].shape}")
    print(f"label:            {item['label']}")
