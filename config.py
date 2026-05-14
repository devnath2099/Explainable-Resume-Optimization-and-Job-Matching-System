from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    # ------------------------------------------------------------------
    # DistilBERT
    # ------------------------------------------------------------------
    model_name: str = "distilbert-base-uncased"
    hidden_size: int = 768

    # ------------------------------------------------------------------
    # BiLSTM
    # ------------------------------------------------------------------
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3

    # ------------------------------------------------------------------
    # Cross-Attention (MultiheadAttention)
    # ------------------------------------------------------------------
    num_attention_heads: int = 8
    attention_dropout: float = 0.1

    # ------------------------------------------------------------------
    # Classification Head
    # ------------------------------------------------------------------
    classifier_dropout: float = 0.3
    classifier_hidden_sizes: List[int] = field(default_factory=lambda: [512, 128])

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    num_classes: int = 3

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    max_length: int = 512
    debug: bool = False
