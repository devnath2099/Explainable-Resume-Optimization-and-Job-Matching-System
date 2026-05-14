# Explainable Resume Optimization & Job Matching System

## Overview

This is the data pipeline component of an end-to-end system that matches resumes to job descriptions using deep learning. The pipeline handles:

- Dataset loading from Hugging Face
- Text cleaning and preprocessing
- PDF resume extraction and parsing
- PyTorch dataset / dataloader preparation

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

If using **Google Colab**, run this in a cell:

```python
!pip install -r requirements.txt
```

### 2. Download NLTK data (optional — handled automatically)

Stopword data is downloaded on first import, but you can force it:

```python
import nltk
nltk.download('stopwords')
```

## Dataset

We use the [cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit) dataset from Hugging Face.

**Columns:**
| Column               | Type   | Description                        |
|----------------------|--------|------------------------------------|
| `resume_text`        | string | Raw resume text                    |
| `job_description_text` | string | Raw job description text         |
| `label`              | int    | 1 = fit, 0 = no fit               |

The dataset is automatically cached by Hugging Face `datasets` after the first load.

## Files

| File               | Purpose                                                     |
|--------------------|-------------------------------------------------------------|
| `dataset.py`       | Loads HF dataset, creates `ResumeJobDataset`, builds loaders |
| `preprocessing.py` | Regex-based text cleaning and normalization                 |
| `pdf_parser.py`    | PDF text extraction (PyMuPDF) and skill/education/experience parsing |
| `utils.py`         | Seed setting, device detection, tokenizer helper             |
| `requirements.txt` | Python dependencies                                          |

## How Preprocessing Works

The `clean_text()` function in `preprocessing.py` applies the following steps **in order**:

1. **HTML artifact removal** — strip `<tags>` and `&entities;`
2. **URL removal** — `https://...` and `www...`
3. **Email removal** — `user@domain.com`
4. **Phone removal** — patterns like `+1 (555) 123-4567`
5. **EEO statement removal** — Equal Opportunity / affirmative action boilerplate
6. **Award boilerplate removal** — generic award phrase patterns
7. **Recruiter signature removal** — "Best regards", "Sincerely", etc.
8. **Whitespace normalization** — collapse multiple spaces / newlines
9. **Lowercasing** — all text to lowercase
10. **Stopword removal** *(optional, off by default)* — filters common English stopwords
11. **Short token filtering** — drops tokens under 2 characters

All removals are regex-based and return cleaned text. The pipeline preserves skills, education, experience, and requirements.

## PDF Parsing

`pdf_parser.py` uses **PyMuPDF (fitz)** to extract text from resume PDFs:

```python
from pdf_parser import parse_resume

result = parse_resume("path/to/resume.pdf")
print(result["skills"])         # list of matched skills
print(result["education"])      # list of education entries
print(result["experience"])     # list of experience entries
print(result["cleaned_text"])   # fully cleaned text
```

## Dataset Pipeline

```python
from dataset import create_dataloaders, get_label_distribution

loaders = create_dataloaders(batch_size=16)

for batch in loaders["train"]:
    # batch["input_ids"]      : torch.Tensor (B, max_len)
    # batch["attention_mask"] : torch.Tensor (B, max_len)
    # batch["label"]          : torch.Tensor (B,)
    print(batch["input_ids"].shape)
    break
```

## Usage Example (end-to-end)

```python
from preprocessing import preprocess_resume, preprocess_job_description
from dataset import create_dataloaders

loaders = create_dataloaders(
    batch_size=16,
    preprocess_fn=preprocess_resume,
)

print(f"Train batches: {len(loaders['train'])}")
print(f"Val batches:   {len(loaders['val'])}")
print(f"Test batches:  {len(loaders['test'])}")
```

## License

MIT
