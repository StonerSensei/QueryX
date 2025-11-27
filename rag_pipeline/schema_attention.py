# rag_pipeline/schema_attention.py
"""
SchemaAttention module

Given:
  - a natural language question
  - a list of schema descriptions (tables/columns)

Returns:
  - a relevance score per schema entry

Used as a fine re-ranking layer on top of vector similarity (Qdrant).
"""

from typing import List

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class SchemaAttention(nn.Module):
    """
    Cross-attention module that scores each schema entry (description)
    against a question using a learned attention head.
    """

    def __init__(self, embed_model_name: str, d_model: int = 384, n_heads: int = 8):
        super().__init__()
        # Underlying SentenceTransformer (can be your fine-tuned queryx-embeddings)
        self.embedder = SentenceTransformer(embed_model_name)

        embed_dim = self.embedder.get_sentence_embedding_dimension()
        self.proj = nn.Linear(embed_dim, d_model)

        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.score_head = nn.Linear(d_model, 1)

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of texts using SentenceTransformer and project to d_model.
        Returns: tensor of shape [len(texts), d_model]
        """
        embs = self.embedder.encode(
            texts,
            convert_to_numpy=False,
            batch_size=32,
            show_progress_bar=False,
        )
        embs = torch.stack(embs, dim=0)  # [N, embed_dim]
        return self.proj(embs)  # [N, d_model]

    def forward(self, question: str, schema_texts: List[str]) -> torch.Tensor:
        """
        Args:
            question: natural-language question string
            schema_texts: list of schema descriptions

        Returns:
            scores: tensor of shape [Ns] (relevance score per schema_text)
        """
        if not schema_texts:
            return torch.zeros(0)

        # Encode question and schema descriptions
        q_emb = self.encode_texts([question])      # [1, d]
        s_emb = self.encode_texts(schema_texts)    # [Ns, d]

        # Add a sequence dimension for attention
        q_emb = q_emb.unsqueeze(1)    # [1, 1, d]
        s_emb = s_emb.unsqueeze(0)    # [1, Ns, d]

        # Cross-attention: query attends over each schema entry
        attn_out, attn_weights = self.attn(q_emb, s_emb, s_emb)
        # attn_weights: [1, 1, Ns] -> attention weight per schema entry

        # Broadcast attention weights back onto schema embeddings
        weights = attn_weights.squeeze(0).squeeze(0)        # [Ns]
        schema_repr = s_emb.squeeze(0) * weights.unsqueeze(-1)  # [Ns, d]

        scores = self.score_head(schema_repr).squeeze(-1)   # [Ns]
        return scores
