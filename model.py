import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Dict, Optional

from config import ModelConfig
from attention import CrossAttention


# ======================================================================
# 1. Shared DistilBERT encoder
# ======================================================================
class DistilBERTEncoder(nn.Module):
    """
    Wraps a pretrained DistilBERT model and returns the last hidden state.
    DistilBERT is shared (same weights) for both resume and job description.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.hidden_size = config.hidden_size

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state


# ======================================================================
# 2. BiLSTM encoder
# ======================================================================
class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM that processes the DistilBERT output sequence.

    Input:  (B, S, D_bert)    D_bert = 768
    Output: (B, S, D_lstm*2)  D_lstm*2 = 512  (bidirectional)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0.0,
        )
        self.output_size = config.lstm_hidden_size * 2

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        output, _ = self.lstm(x)
        return output


# ======================================================================
# 3. Classification Head
# ======================================================================
class ClassificationHead(nn.Module):
    """
    MLP classifier with dropout after each hidden layer.
    Input:  (B, D)
    Output: (B, num_classes)
    """

    def __init__(self, config: ModelConfig, input_size: int):
        super().__init__()
        layers = []
        prev = input_size
        for hidden in config.classifier_hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.classifier_dropout))
            prev = hidden
        layers.append(nn.Linear(prev, config.num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ======================================================================
# 4. Full Model
# ======================================================================
class ResumeJobMatcher(nn.Module):
    """
    End-to-end architecture for resume-job description matching.

    Architecture:
        Resume Text  ──► DistilBERT ──► BiLSTM ──┐
                                                  ├──► Cross-Attention ──► Pooling ──► Dense ──► Score
        Job Text     ──► DistilBERT ──► BiLSTM ──┘
                     (shared weights)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Shared text encoders
        self.bert_encoder = DistilBERTEncoder(config)
        self.bilstm = BiLSTMEncoder(config)

        lstm_out = config.lstm_hidden_size * 2  # 512

        # Projection to match attention dimension (identity if same dim)
        self.projection = nn.Linear(lstm_out, lstm_out)

        # Cross-attention
        self.cross_attention = CrossAttention(
            hidden_size=lstm_out,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
        )

        # Projection after cross-attention
        self.attn_projection = nn.Linear(lstm_out * 2, lstm_out)

        # Classification head
        self.classifier = ClassificationHead(config, input_size=lstm_out)

    def forward(
        self,
        resume_input_ids: torch.Tensor,
        resume_mask: torch.Tensor,
        job_input_ids: torch.Tensor,
        job_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            resume_input_ids: (B, R_max_len)
            resume_mask:      (B, R_max_len)
            job_input_ids:    (B, J_max_len)
            job_mask:         (B, J_max_len)

        Returns:
            dict with:
                logits:              (B, num_classes)
                probabilities:       (B, num_classes)
                attention_weights:   dict with resume→job and job→resume
                pooled_embeddings:   (B, lstm_out)
        """
        # ------------------------------------------------------------------
        # Step 1: Shared DistilBERT encoding
        # ------------------------------------------------------------------
        resume_bert = self.bert_encoder(resume_input_ids, resume_mask)  # (B, R, 768)
        job_bert = self.bert_encoder(job_input_ids, job_mask)           # (B, J, 768)

        # ------------------------------------------------------------------
        # Step 2: BiLSTM
        # ------------------------------------------------------------------
        resume_lstm = self.bilstm(resume_bert, resume_mask)  # (B, R, 512)
        job_lstm = self.bilstm(job_bert, job_mask)           # (B, J, 512)

        # ------------------------------------------------------------------
        # Step 3: Project to uniform dimension
        # ------------------------------------------------------------------
        resume_embeds = self.projection(resume_lstm)  # (B, R, 512)
        job_embeds = self.projection(job_lstm)        # (B, J, 512)

        # ------------------------------------------------------------------
        # Step 4: Cross-attention
        # ------------------------------------------------------------------
        attn_out = self.cross_attention(
            resume_embeds=resume_embeds,
            job_embeds=job_embeds,
            resume_mask=resume_mask,
            job_mask=job_mask,
        )
        resume_attended = attn_out["resume_attended"]  # (B, R, 512)
        job_attended = attn_out["job_attended"]        # (B, J, 512)

        # ------------------------------------------------------------------
        # Step 5: Mean pooling (masked)
        # ------------------------------------------------------------------
        resume_pooled = self._mean_pooling(resume_attended, resume_mask)  # (B, 512)
        job_pooled = self._mean_pooling(job_attended, job_mask)           # (B, 512)

        combined = torch.cat([resume_pooled, job_pooled], dim=-1)  # (B, 1024)
        pooled = self.attn_projection(combined)                    # (B, 512)

        # ------------------------------------------------------------------
        # Step 6: Classification head
        # ------------------------------------------------------------------
        logits = self.classifier(pooled)                     # (B, num_classes)
        probs = F.softmax(logits, dim=-1)                    # (B, num_classes)

        # ------------------------------------------------------------------
        # Debug prints
        # ------------------------------------------------------------------
        if self.config.debug:
            print(f"[DEBUG] resume_bert:       {list(resume_bert.shape)}")
            print(f"[DEBUG] job_bert:          {list(job_bert.shape)}")
            print(f"[DEBUG] resume_lstm:       {list(resume_lstm.shape)}")
            print(f"[DEBUG] job_lstm:          {list(job_lstm.shape)}")
            print(f"[DEBUG] resume_attended:   {list(resume_attended.shape)}")
            print(f"[DEBUG] job_attended:      {list(job_attended.shape)}")
            print(f"[DEBUG] resume_pooled:     {list(resume_pooled.shape)}")
            print(f"[DEBUG] job_pooled:        {list(job_pooled.shape)}")
            print(f"[DEBUG] combined:          {list(combined.shape)}")
            print(f"[DEBUG] pooled:            {list(pooled.shape)}")
            print(f"[DEBUG] logits:            {list(logits.shape)}")
            print(f"[DEBUG] probs:             {list(probs.shape)}")

        return {
            "logits": logits,
            "probabilities": probs,
            "attention_weights": {
                "resume_to_job": attn_out["resume_attention_weights"],
                "job_to_resume": attn_out["job_attention_weights"],
            },
            "pooled_embeddings": pooled,
        }

    # ------------------------------------------------------------------
    # Masked mean pooling helper
    # ------------------------------------------------------------------
    @staticmethod
    def _mean_pooling(
        hidden_states: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        hidden_states: (B, S, D)
        mask:          (B, S)     — 1 = real token, 0 = padding
        Returns:       (B, D)
        """
        mask_expanded = mask.unsqueeze(-1).float()           # (B, S, 1)
        masked = hidden_states * mask_expanded
        summed = masked.sum(dim=1)                           # (B, D)
        lengths = mask.sum(dim=1, keepdim=True).float().clamp(min=1)  # (B, 1)
        return summed / lengths


# ======================================================================
# Convenience: instantiate model and move to device
# ======================================================================
def build_model(
    config: Optional[ModelConfig] = None,
    device: Optional[torch.device] = None,
) -> ResumeJobMatcher:
    if config is None:
        config = ModelConfig()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResumeJobMatcher(config).to(device)
    print(f"[INFO] Model built on {device}")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {total:,}")
    return model


# ======================================================================
# Quick smoke test
# ======================================================================
if __name__ == "__main__":
    config = ModelConfig(debug=True)
    model = build_model(config)

    B, R, J = 2, 64, 48
    resume_ids = torch.randint(0, 30522, (B, R))
    resume_mask = torch.ones(B, R, dtype=torch.long)
    job_ids = torch.randint(0, 30522, (B, J))
    job_mask = torch.ones(B, J, dtype=torch.long)

    out = model(resume_ids, resume_mask, job_ids, job_mask)
    print(f"\nlogits:       {out['logits'].shape}")
    print(f"probs:        {out['probabilities'].shape}")
    print(f"r2j weights:  {out['attention_weights']['resume_to_job'].shape}")
    print(f"j2r weights:  {out['attention_weights']['job_to_resume'].shape}")
    print(f"pooled:       {out['pooled_embeddings'].shape}")
