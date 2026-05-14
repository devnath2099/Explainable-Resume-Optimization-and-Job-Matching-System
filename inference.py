import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import Dict, Optional
from pathlib import Path

from config import ModelConfig
from model import ResumeJobMatcher
from utils_train import ID_TO_LABEL
from pdf_parser import extract_resume_text, clean_resume_text

THRESHOLDS = {"low": 0.4, "high": 0.7}
DEFAULT_MODEL_PATH = "saved_models/best_model.pt"


def load_model(
    model_path: str = DEFAULT_MODEL_PATH,
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
    print(f"[INFO] Best val F1: {checkpoint.get('val_f1', 'N/A'):.4f}" if "val_f1" in checkpoint else "")
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
    fit_prob = float(probs[1].item())
    pred_class = int(logits.argmax(dim=-1).item())
    probs_list = [round(float(p), 4) for p in probs.cpu().tolist()]

    interpretation = interpret_score(fit_prob)

    return {
        "fit_probability": round(fit_prob, 4),
        "prediction": interpretation["label"],
        "class_id": pred_class,
        "probabilities": probs_list,
        "interpretation": interpretation,
    }


def predict_pdf(
    model: ResumeJobMatcher,
    pdf_path: str,
    job_text: str,
    tokenizer,
    device: Optional[torch.device] = None,
    max_length: int = 128,
) -> Dict:
    raw_text = extract_resume_text(pdf_path)
    resume_text = clean_resume_text(raw_text)
    return predict(model, resume_text, job_text, tokenizer, device, max_length)


def interpret_score(fit_prob: float) -> Dict:
    if fit_prob < THRESHOLDS["low"]:
        return {"label": "No Fit", "threshold": f"<{THRESHOLDS['low']}", "severity": "low"}
    elif fit_prob < THRESHOLDS["high"]:
        return {"label": "Potential Fit", "threshold": f"{THRESHOLDS['low']}-{THRESHOLDS['high']}", "severity": "medium"}
    else:
        return {"label": "Strong Fit", "threshold": f">{THRESHOLDS['high']}", "severity": "high"}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    model_path = Path(DEFAULT_MODEL_PATH)
    if model_path.exists():
        model = load_model(str(model_path), device)
    else:
        print(f"[WARN] No saved model found at {model_path}. Using random weights.")
        model = ResumeJobMatcher(ModelConfig(debug=False)).to(device)
        model.eval()

    result = predict(
        model=model,
        resume_text="Python developer with 5 years of experience in Django, AWS, and PostgreSQL. "
                     "Built REST APIs and deployed microservices on Kubernetes.",
        job_text="Looking for a senior backend engineer with Python, Django, and cloud experience.",
        tokenizer=tokenizer,
        device=device,
        max_length=128,
    )

    for k, v in result.items():
        print(f"  {k}: {v}")
