import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import compute_metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device("cpu"),
        gradient_clip_norm: float = 1.0,
        save_dir: str = "saved_models",
        log_interval: int = 10,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip_norm = gradient_clip_norm
        self.save_dir = save_dir
        self.log_interval = log_interval

        self.criterion = nn.CrossEntropyLoss()
        self.best_val_f1 = 0.0

        os.makedirs(self.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # One training epoch
    # ------------------------------------------------------------------
    def train_epoch(self) -> tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        pbar = tqdm(self.train_loader, desc="  Train", leave=False)
        for batch_idx, batch in enumerate(pbar):
            resume_ids = batch["resume_input_ids"].to(self.device)
            resume_mask = batch["resume_mask"].to(self.device)
            job_ids = batch["job_input_ids"].to(self.device)
            job_mask = batch["job_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                resume_input_ids=resume_ids,
                resume_mask=resume_mask,
                job_input_ids=job_ids,
                job_mask=job_mask,
            )

            loss = self.criterion(outputs["logits"], labels)
            loss.backward()

            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm
            )

            self.optimizer.step()

            total_loss += loss.item()
            preds = outputs["logits"].argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        metrics = compute_metrics(all_preds, all_labels)
        return avg_loss, metrics

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self) -> tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []

        pbar = tqdm(self.val_loader, desc="  Val", leave=False)
        for batch in pbar:
            resume_ids = batch["resume_input_ids"].to(self.device)
            resume_mask = batch["resume_mask"].to(self.device)
            job_ids = batch["job_input_ids"].to(self.device)
            job_mask = batch["job_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            outputs = self.model(
                resume_input_ids=resume_ids,
                resume_mask=resume_mask,
                job_input_ids=job_ids,
                job_mask=job_mask,
            )

            loss = self.criterion(outputs["logits"], labels)
            total_loss += loss.item()

            preds = outputs["logits"].argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / max(len(self.val_loader), 1)
        metrics = compute_metrics(all_preds, all_labels)
        return avg_loss, metrics

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------
    def fit(self, num_epochs: int) -> float:
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*54}")
            print(f"  Epoch {epoch}/{num_epochs}")
            print(f"{'='*54}")

            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics = self.validate()

            if self.scheduler is not None:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"  LR: {lr:.2e}  |  "
                f"Train Loss: {train_loss:.4f}  |  "
                f"Val Loss: {val_loss:.4f}"
            )
            print(
                f"  Train Acc: {train_metrics['accuracy']:.4f}  |  "
                f"Train F1: {train_metrics['f1']:.4f}"
            )
            print(
                f"  Val   Acc: {val_metrics['accuracy']:.4f}  |  "
                f"Val   F1: {val_metrics['f1']:.4f}"
            )

            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self._save_checkpoint(epoch, val_metrics)
                print(f"  \u2713 Best model saved (F1: {val_metrics['f1']:.4f})")

        print(f"\n{'='*54}")
        print(f"  Training complete. Best val F1: {self.best_val_f1:.4f}")
        print(f"{'='*54}")
        return self.best_val_f1

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> str:
        path = os.path.join(self.save_dir, "best_model.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_f1": metrics["f1"],
                "val_accuracy": metrics["accuracy"],
            },
            path,
        )
        return path


if __name__ == "__main__":
    from config import ModelConfig
    from model import ResumeJobMatcher
    from utils_train import SeparateTokenizationDataset, TrainingConfig
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    ds = SeparateTokenizationDataset(
        resume_texts=["Python developer with 5 years experience"] * 8,
        job_texts=["Senior Python engineer needed"] * 8,
        labels=[1] * 8,
        tokenizer=tokenizer,
        max_length=32,
    )
    loader = DataLoader(ds, batch_size=4)

    model = ResumeJobMatcher(ModelConfig(debug=False)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=optim,
        device=device,
    )

    val_loss, val_metrics = trainer.validate()
    print(f"Smoke test - val loss: {val_loss:.4f}, f1: {val_metrics['f1']:.4f}")
    print("Trainer smoke test passed.")
