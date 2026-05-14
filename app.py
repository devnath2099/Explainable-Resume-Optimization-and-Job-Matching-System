from pathlib import Path

import gradio as gr
import torch
from transformers import AutoTokenizer

from inference import (
    load_model,
    predict_pdf,
    DEFAULT_MODEL_PATH,
)
from explainability import (
    get_matched_skills,
    get_missing_skills,
    get_redundant_skills,
    generate_explanation,
    generate_suggestions,
)
from pdf_parser import extract_resume_text, clean_resume_text

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model_path = Path(DEFAULT_MODEL_PATH)
if model_path.exists():
    MODEL = load_model(str(model_path), DEVICE)
    print("[INFO] Model ready for inference.")
else:
    print(f"[ERROR] No trained model found at {DEFAULT_MODEL_PATH}.")
    MODEL = None


def predict_fn(pdf_file, job_text):
    if MODEL is None:
        return "Model not loaded. Train first.", None, "", [], [], []

    if pdf_file is None:
        return "Please upload a resume PDF.", None, "", [], [], []

    if not job_text or not job_text.strip():
        return "Please enter a job description.", None, "", [], [], []

    try:
        result = predict_pdf(MODEL, pdf_file.name, job_text, TOKENIZER, DEVICE)
        fit_prob = result["fit_probability"]
        label = result["interpretation"]["label"]
        prob_display = f"{fit_prob:.1%}" if fit_prob > 1 else f"{fit_prob:.2f}"

        raw_text = extract_resume_text(pdf_file.name)
        resume_clean = clean_resume_text(raw_text)

        matched = get_matched_skills(resume_clean, job_text)
        missing = get_missing_skills(resume_clean, job_text)
        redundant = get_redundant_skills(resume_clean, job_text)

        explanation = generate_explanation(matched, missing, redundant, fit_prob, result["interpretation"])
        suggestions = generate_suggestions(missing, redundant, fit_prob)

        return f"{label} ({prob_display})", fit_prob, explanation, matched, missing, suggestions

    except Exception as e:
        return f"Error: {str(e)}", None, "", [], [], []


with gr.Blocks(
    title="Resume-Job Matching System",
    theme=gr.themes.Soft(),
    css="footer {visibility: hidden}",
) as demo:
    gr.Markdown(
        """
        # Resume-Job Matching System
        Upload a resume PDF and enter a job description to assess fit.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="Upload Resume (PDF)",
                file_types=[".pdf"],
                type="filepath",
            )
            jd_input = gr.Textbox(
                label="Job Description",
                placeholder="Paste job description here...",
                lines=8,
            )
            predict_btn = gr.Button("Predict Fit", variant="primary", size="lg")

        with gr.Column(scale=1):
            prediction_output = gr.Textbox(label="Prediction", interactive=False)
            prob_output = gr.Number(label="Fit Score", interactive=False)

            with gr.Accordion("Explanation", open=True):
                explanation_output = gr.Markdown()

            with gr.Accordion("Matched Skills", open=True):
                matched_output = gr.JSON()

            with gr.Accordion("Missing Skills", open=True):
                missing_output = gr.JSON()

            with gr.Accordion("Optimization Suggestions", open=True):
                suggestions_output = gr.JSON()

    predict_btn.click(
        fn=predict_fn,
        inputs=[pdf_input, jd_input],
        outputs=[
            prediction_output,
            prob_output,
            explanation_output,
            matched_output,
            missing_output,
            suggestions_output,
        ],
    )

    gr.Markdown(
        """
        ---
        **Interpretation Guide:**
        - **Fit Score < 0.4** → No Fit
        - **Fit Score 0.4 – 0.7** → Potential Fit
        - **Fit Score > 0.7** → Strong Fit
        """
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public Colab link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0",
    )
