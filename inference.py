import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import Dict, Optional

from config import ModelConfig
from model import ResumeJobMatcher
from utils_train import ID_TO_LABEL


def load_model(
    model_path: str,
    device: Optional[torch.device] = None,
) -> ResumeJobMatcher:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ModelConfig(debug=False)
    model = ResumeJobMatcher(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[INFO] Model loaded from {model_path} on {device}")
    return model


def predict(
    model: ResumeJobMatcher,
    resume_text: str,
    job_text: str,
    tokenizer,
    device: Optional[torch.device] = None,
    max_length: int = 128,
) -> Dict:
    if device is None:
        device = next(model.parameters()).device

    resume_enc = tokenizer(
        resume_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    job_enc = tokenizer(
        job_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            resume_input_ids=resume_enc["input_ids"].to(device),
            resume_mask=resume_enc["attention_mask"].to(device),
            job_input_ids=job_enc["input_ids"].to(device),
            job_mask=job_enc["attention_mask"].to(device),
        )

    logits = outputs["logits"]
    probs = F.softmax(logits, dim=-1).squeeze(0)
    pred_class = int(logits.argmax(dim=-1).item())
    confidence = float(probs[pred_class].item())
    probs_list = [round(float(p), 4) for p in probs.cpu().tolist()]

    return {
        "prediction": ID_TO_LABEL[pred_class],
        "class_id": pred_class,
        "confidence": round(confidence, 4),
        "probabilities": probs_list,
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Build a fresh model with random weights (for testing only)
    model = ResumeJobMatcher(ModelConfig(debug=False)).to(device)
    model.eval()

    result = predict(
        model=model,
        resume_text="Python developer with 5 years of experience in Django and AWS.",
        job_text="Looking for a senior backend engineer with Python and cloud experience.",
        tokenizer=tokenizer,
        device=device,
        max_length=64,
    )

    for k, v in result.items():
        print(f"  {k}: {v}")
