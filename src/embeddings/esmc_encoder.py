# scr/embeddings/esm_encoder.py
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ── CONFIGURATION ─────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = Path("../../temp_model/esm2")

# Load tokenizer and model once at import
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
_MODEL = AutoModel.from_pretrained(MODEL_DIR).to(DEVICE).eval()

# PCA cache globals
global _PCA, _MAX_LEN
_PCA: IncrementalPCA | None = None
_MAX_LEN: int | None = None


def _save_pca(path: Path, pca: IncrementalPCA, max_len: int) -> None:
   with open(path, "wb") as fh:
       pickle.dump({"pca": pca, "max_len": max_len}, fh)


def _load_pca(path: Path) -> tuple[IncrementalPCA, int]:
   with open(path, "rb") as fh:
       obj: dict[str, Any] = pickle.load(fh)
   return obj["pca"], obj["max_len"]


@torch.inference_mode()
def _embed(seq: str) -> np.ndarray:
   """
   Tokenize and get ESM-2 last_hidden_state embeddings for a single sequence.
   Returns a NumPy array of shape (L, D_model).
   """
   enc = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
   input_ids = enc["input_ids"].to(DEVICE)
   attention_mask = enc["attention_mask"].to(DEVICE)
   outputs = _MODEL(input_ids=input_ids, attention_mask=attention_mask)
   reps = outputs.last_hidden_state[0]  # shape = (L, D_model)
   return reps.cpu().numpy()


def embed_sequences(
   df: pd.DataFrame,
   seq_col: str = "sequence",
   n_components: int = 1024,
   pca_batch: int = 1024,
   save_dir: str | Path | None = None,
) -> pd.DataFrame:
   """
   Add a `pca_embed` column to df by:
     1. Tokenizing+embedding each padded sequence with ESM-2 (L×D_model)
     2. Flattening to length L*D_model
     3. Training (or loading) an IncrementalPCA to reduce to n_components
   """
   save_dir = Path(save_dir or "../data/processed")
   save_dir.mkdir(parents=True, exist_ok=True)
   pca_path = save_dir / f"pca_{n_components}.pkl"

   sequences = df[seq_col].astype(str).tolist()

   global _PCA, _MAX_LEN
   # 1) train or load PCA
   if pca_path.exists():
       _PCA, _MAX_LEN = _load_pca(pca_path)
   else:
       _MAX_LEN = max(len(s) for s in sequences)
       _PCA = IncrementalPCA(n_components=n_components, batch_size=pca_batch)
       flat_vecs: list[np.ndarray] = []
       for s in tqdm(sequences, desc="ESM-2 embedding (train PCA)"):
           padded = s.ljust(_MAX_LEN, "X")
           token_emb = _embed(padded)
           flat_vecs.append(token_emb.reshape(-1))
       X = np.stack(flat_vecs)
       for i in range(0, len(X), pca_batch):
           _PCA.partial_fit(X[i : i + pca_batch])
       _save_pca(pca_path, _PCA, _MAX_LEN)

   # 2) embed + apply PCA
   flat_vecs: list[np.ndarray] = []
   for s in tqdm(sequences, desc="ESM-2 embedding (apply PCA)"):
       padded = s.ljust(_MAX_LEN, "X")
       token_emb = _embed(padded)
       flat_vecs.append(token_emb.reshape(-1))

   X_full = np.stack(flat_vecs)
   X_reduced = _PCA.transform(X_full)

   out = df.copy()
   out["pca_embed"] = X_reduced.tolist()
   return out


def embed_single(
   seq: str,
   n_components: int = 1024,
   pca_dir: str | Path = "../data/processed",
   pad_length: int | None = None,
) -> np.ndarray:
   """
   Embed a single sequence and project via cached PCA.
   """
   global _PCA, _MAX_LEN
   if _PCA is None or _MAX_LEN is None:
       pca_path = Path(pca_dir) / f"pca_{n_components}.pkl"
       if not pca_path.exists():
           raise RuntimeError(f"No PCA found at {pca_path}. Run embed_sequences first.")
       _PCA, _MAX_LEN = _load_pca(pca_path)

   final_len = pad_length or _MAX_LEN
   padded = str(seq).ljust(final_len, "X")
   token_emb = _embed(padded)
   flat = token_emb.reshape(1, -1)
   return _PCA.transform(flat).squeeze(0)