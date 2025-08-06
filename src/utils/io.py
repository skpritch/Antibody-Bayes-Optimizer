# src/utils/io.py

from __future__ import annotations
import pandas as pd, json, gzip
from typing import Sequence
from pathlib import Path
import numpy as np
import pandas as pd
from embeddings.esmc_encoder import embed_sequences


def make_embedding_lookup(df: pd.DataFrame, embed_col: str, cfg, interim_dir: Path,) -> callable:
  """
  Build and return get_embedding(seq) → 1×D numpy array.
   - Preloads existing embeddings from df[embed_col].
  - For any new seq, calls embed_sequences(...) to compute and caches it.
  """
  seq2emb: dict[str, np.ndarray] = {}
  for row in df.itertuples():
      emb = getattr(row, embed_col, None)
      # skip if missing or NaN
      if emb is None:
          continue
      # emb might be a float nan if missing
      if isinstance(emb, float) and np.isnan(emb):
          continue
      # otherwise assume list‑like
      arr = np.array(emb, dtype=np.float32)
      seq2emb[row.sequence] = arr


  def get_embedding(seq: str) -> np.ndarray:
      # return cached if present
      if seq in seq2emb:
          return seq2emb[seq]
      # else compute fresh
      df_tmp = pd.DataFrame({"sequence": [seq]})
      df_red = embed_sequences(
          df_tmp,
          n_components=cfg.embed_components,
          save_dir=interim_dir
      )
      new_emb = df_red[embed_col].iloc[0]
      arr = np.array(new_emb, dtype=np.float32)
      seq2emb[seq] = arr
      return arr

  return get_embedding


def read(path: str | Path):
  p = Path(path)
  if p.suffix == ".json":
      with open(p) as f:
          df = pd.DataFrame(json.loads(line) for line in f)
  elif p.suffix in {".gz", ".gzip"}:
      with gzip.open(p, "rt") as f:
          df = pd.read_json(f, lines=True)
  else:
      df = pd.read_csv(p)
  return df


def write_jsonl(df: pd.DataFrame, path: str | Path, columns: Sequence[str] | None = None):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w") as f:
      for rec in df[columns] if columns else df.to_dict(orient="records"):
          f.write(json.dumps(rec, default=str) + "\n")