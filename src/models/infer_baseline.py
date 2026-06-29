"""Baseline FinBERT inference: score chunk texts into per-class probabilities.

``score_batch`` is the core unit: it tokenizes a batch of chunk strings, runs a
forward pass under ``torch.no_grad()``, and returns softmax probabilities as a
plain NumPy array shaped ``(n_texts, n_classes)``.

``get_device`` and ``load_model`` are thin loaders kept deliberately minimal;
the model is the pretrained FinBERT (``yiyanghkust/finbert-pretrain``) with a
3-class sequence-classification head.
"""

import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizer

MODEL_NAME = "yiyanghkust/finbert-pretrain"
NUM_LABELS = 3
MAX_LENGTH = 512


def get_device() -> torch.device:
    """Return the CUDA device (inference runs on Colab with a GPU)."""
    return torch.device("cuda")


def load_model(model_name: str = MODEL_NAME):
    """Load the FinBERT tokenizer and 3-class sequence-classification model."""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS
    )
    model.eval()
    return tokenizer, model


def score_batch(texts: list[str], tokenizer, model, device) -> np.ndarray:
    """Return softmax class probabilities for a batch of texts.

    Texts are tokenized with padding/truncation to ``MAX_LENGTH`` tokens and run
    through ``model`` on ``device`` under ``torch.no_grad()``. The logits are
    softmaxed over the class dimension and returned as a NumPy array of shape
    ``(len(texts), NUM_LABELS)``.
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    return probs.cpu().numpy()
