import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class CrossAttention(nn.Module):
    """
    Bidirectional cross-attention between resume and job description.

    Architecture:
        resume_embeds  ──┐
                         ├──► Resume→Job attention
        job_embeds ──────┘
                         ┌──► Job→Resume attention
        resume_embeds ──┤
        job_embeds    ──┘

    Each uses nn.MultiheadAttention where the query attends to the other
    modality's key/value pairs.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        resume_embeds: torch.Tensor,
        job_embeds: torch.Tensor,
        resume_mask: Optional[torch.Tensor] = None,
        job_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            resume_embeds: (B, R, D)  — resume sequence embeddings
            job_embeds:    (B, J, D)  — job description sequence embeddings
            resume_mask:   (B, R)     — 1 = real token, 0 = padding
            job_mask:      (B, J)     — 1 = real token, 0 = padding

        Returns:
            dict with:
                resume_attended:       (B, R, D)
                job_attended:          (B, J, D)
                resume_attention_weights: (B, R, J)   avg over heads
                job_attention_weights:    (B, J, R)   avg over heads
        """
        # --- Resume attends to Job ---
        r_key_padding = (job_mask == 0) if job_mask is not None else None
        resume_attended, r_weights = self.attention(
            query=resume_embeds,
            key=job_embeds,
            value=job_embeds,
            key_padding_mask=r_key_padding,
            need_weights=True,
            average_attn_weights=True,
        )

        # --- Job attends to Resume ---
        j_key_padding = (resume_mask == 0) if resume_mask is not None else None
        job_attended, j_weights = self.attention(
            query=job_embeds,
            key=resume_embeds,
            value=resume_embeds,
            key_padding_mask=j_key_padding,
            need_weights=True,
            average_attn_weights=True,
        )

        return {
            "resume_attended": resume_attended,
            "job_attended": job_attended,
            "resume_attention_weights": r_weights,
            "job_attention_weights": j_weights,
        }
