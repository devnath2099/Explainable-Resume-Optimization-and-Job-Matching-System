import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

DATASET_NAME = "cnamuangtoun/resume-job-description-fit"
DEFAULT_MODEL_NAME = "bert-base-uncased"
LABEL_MAP = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}


class ResumeJobDataset(Dataset):
    def __init__(
        self,
        resume_texts: List[str],
        job_texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512,
        combine_text: bool = True,
    ):
        self.resume_texts = resume_texts
        self.job_texts = job_texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.combine_text = combine_text

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        resume = str(self.resume_texts[idx])
        job = str(self.job_texts[idx])

        if self.combine_text:
            text = f"{resume} [SEP] {job}"
        else:
            text = resume + " " + job

        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_raw_dataset(
    dataset_name: str = DATASET_NAME,
) -> Dict:
    dataset = load_dataset(dataset_name)
    return dataset


def prepare_data_splits(
    dataset_name: str = DATASET_NAME,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    preprocess_fn: Optional[Callable] = None,
) -> Dict[str, Dict]:

    raw = load_raw_dataset(dataset_name)
    train_data = raw["train"]

    resume_texts: List[str] = train_data["resume_text"]
    job_texts: List[str] = train_data["job_description_text"]
    raw_labels = train_data["label"]
    labels: List[int] = [LABEL_MAP[l] for l in raw_labels]

    if preprocess_fn:
        resume_texts = [preprocess_fn(t) for t in resume_texts]
        job_texts = [preprocess_fn(t) for t in job_texts]

    indices = list(range(len(labels)))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(test_size + val_size),
        random_state=random_state,
        stratify=labels,
    )

    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_size / (test_size + val_size),
        random_state=random_state,
        stratify=temp_labels,
    )

    splits: Dict[str, Dict] = {}
    for name, idx_list in [
        ("train", train_idx),
        ("val", val_idx),
        ("test", test_idx),
    ]:
        splits[name] = {
            "resume_texts": [resume_texts[i] for i in idx_list],
            "job_texts": [job_texts[i] for i in idx_list],
            "labels": [labels[i] for i in idx_list],
        }

    return splits


def create_dataloaders(
    batch_size: int = 16,
    max_length: int = 512,
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_name: str = DATASET_NAME,
    preprocess_fn: Optional[Callable] = None,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    splits = prepare_data_splits(
        dataset_name=dataset_name,
        preprocess_fn=preprocess_fn,
    )

    datasets: Dict[str, ResumeJobDataset] = {}
    for split_name, data in splits.items():
        datasets[split_name] = ResumeJobDataset(
            resume_texts=data["resume_texts"],
            job_texts=data["job_texts"],
            labels=data["labels"],
            tokenizer=tokenizer,
            max_length=max_length,
        )

    loaders: Dict[str, DataLoader] = {}
    for split_name, dataset in datasets.items():
        shuffle = split_name == "train"
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders


def get_label_distribution(labels: List[int]) -> Dict[str, float]:
    total = len(labels)
    if total == 0:
        return {}
    pos = sum(labels)
    neg = total - pos
    return {
        "total": total,
        "positive": pos,
        "negative": neg,
        "positive_ratio": round(pos / total, 4),
        "negative_ratio": round(neg / total, 4),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loaders = create_dataloaders(batch_size=8)
    for split_name, loader in loaders.items():
        batch = next(iter(loader))
        print(f"{split_name}: input_ids shape = {batch['input_ids'].shape}, "
              f"labels shape = {batch['label'].shape}")

    logger.info("Dataset pipeline ready.")
