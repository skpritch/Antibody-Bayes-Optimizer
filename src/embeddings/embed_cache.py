# src/embeddings/embed_cache.py

from __future__ import annotations
import numpy as np
from functools import lru_cache
from embeddings.esm_encoder import embed_single


def get_pad_length(seqs: list[str]) -> int:
  """Return max length so we can pad consistently with 'X'."""
  return max(len(s) for s in seqs)

def build_embed_cache(n_components: int, pad_length: int):
  """Returns a cached embedding function for a fixed pad length."""

  @lru_cache(maxsize=200_000)
  def _embed(seq: str) -> np.ndarray:
      return embed_single(seq, n_components, pad_length=pad_length)

  return _embed

def build_fitness_cache(embed_fn, gp):
  """Vectorised predictor & single‑seq fitness cache."""

  @lru_cache(maxsize=200_000)
  def predict_one(seq: str) -> float:
      emb = embed_fn(seq).reshape(1, -1)
      return float(gp.predict(emb)[0])

  def predict_many(seqs: list[str]) -> np.ndarray:
      embs = np.vstack([embed_fn(s) for s in seqs])
      return gp.predict(embs)

  return predict_one, predict_many