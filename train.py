import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import ModelConfig
from dataset import prepare_data_splits
from model import ResumeJobMatcher, build_model
from trainer import Trainer, EarlyStopping
from utils_train import (
    SeparateTokenizationDataset,
    TrainingConfig,
    seed_everything,
)


def create_train_val_loaders(
    train_cfg: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.model_name)

    print("[INFO] Loading dataset splits ...")
    splits = prepare_data_splits(random_state=train_cfg.seed)

    train_ds = SeparateTokenizationDataset(
        resume_texts=splits["train"]["resume_texts"],
        job_texts=splits["train"]["job_texts"],
        labels=splits["train"]["labels"],
        tokenizer=tokenizer,
        max_length=train_cfg.max_seq_length,
    )
    val_ds = SeparateTokenizationDataset(
        resume_texts=splits["val"]["resume_texts"],
        job_texts=splits["val"]["job_texts"],
        labels=splits["val"]["labels"],
        tokenizer=tokenizer,
        max_length=train_cfg.max_seq_length,
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )

    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val   samples: {len(val_ds)}")
    print(f"  Batch size:    {train_cfg.batch_size}")
    print(f"  Max seq len:   {train_cfg.max_seq_length}")
    return train_loader, val_loader


def main() -> None:
    train_cfg = TrainingConfig()
    model_cfg = ModelConfig()

    seed_everything(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader = create_train_val_loaders(train_cfg)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(model_cfg, device)

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    early_stopping = EarlyStopping(patience=train_cfg.early_stopping_patience)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        device=device,
        gradient_clip_norm=train_cfg.gradient_clip_norm,
        save_dir=train_cfg.save_dir,
        log_interval=train_cfg.log_interval,
    )

    trainer.fit(num_epochs=train_cfg.num_epochs)

    print("\n[DONE] Training complete.")


if __name__ == "__main__":
    main()
