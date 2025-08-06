# src/utils/utils.py

from __future__ import annotations
import numpy as np
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Callable

from developability.dev_score import score_sequences

def toy_ground_truth(x: np.ndarray) -> np.ndarray:
  """
  Simple multimodal surface in [0,1]^D.
  Parameters
  ----------
  x : (N,D) ndarray
  Returns
  -------
  (N,) ndarray
  """
  return np.sum((x - 0.5) ** 2, axis=1) - np.sin(5 * np.sum(x, axis=1))


def load_or_compute_embeddings(
  raw_json: str | Path,
  interim_file: str | Path,
  seq_col: str,
  embed_col: str,
  embed_fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
  """
  Load/calc ESM‑C+PCA embeddings with caching.

  raw_json      : path to input JSONL with at least `seq_col`
  interim_file  : path to cache JSONL with embeddings
  seq_col       : name of sequence column
  embed_col     : name of embedding column to fill
  embed_fn      : fn(df_subset) -> df_with_embed_col
  """
  raw_json     = Path(raw_json)
  interim_file = Path(interim_file)

  # 1) load existing or raw
  if interim_file.exists():
      df = pd.read_json(interim_file, lines=True)
  else:
      df = pd.read_json(raw_json, lines=True)

  # 2) mask missing
  if embed_col in df:
      mask = df[embed_col].isna()
  else:
      df[embed_col] = np.nan
      mask = pd.Series(True, index=df.index)

  # 3) compute missing
  if mask.any():
      df_sub = df.loc[mask, [seq_col]].copy()
      df_emb = embed_fn(df_sub)
      df.loc[mask, embed_col] = df_emb.loc[mask, embed_col].values

  # 4) save full cache
  df.to_json(interim_file, orient="records", lines=True)
  print(f"Embeddings ready: {df[embed_col].notna().sum()} / {len(df)}")
  return df


def load_or_compute_developability(
   interim_embed_file: str | Path,
   interim_dev_file:   str | Path,
   seq_col:            str,
   dev_col:            str,
   score_fn:           Callable[..., List[float]],
   dev_json_dir:       str | Path = "../data/biophi",
   seq_id_col:         str       = "seq_id"
) -> pd.DataFrame:
   interim_embed_file = Path(interim_embed_file)
   interim_dev_file   = Path(interim_dev_file)
   dev_json_dir       = Path(dev_json_dir)
   dev_json_dir.mkdir(parents=True, exist_ok=True)

   # 1) load existing cache or embeddings
   if interim_dev_file.exists():
       df = pd.read_json(interim_dev_file, lines=True)
   else:
       df = pd.read_json(interim_embed_file, lines=True)

   # 1b) ensure seq_id column exists
   if seq_id_col not in df.columns:
       df[seq_id_col] = list(range(len(df)))

   # 2) initialize dev_col if missing
   if dev_col not in df.columns:
       df[dev_col] = np.nan

   # 3) compute missing scores one by one
   missing_idxs = df.index[df[dev_col].isna()]
   for idx in missing_idxs:
       seq = df.at[idx, seq_col]
       sid = int(df.at[idx, seq_id_col])
       try:
           # score_fn expects lists of sequences and IDs
           score = score_fn([seq], [sid], dev_dir=dev_json_dir)[0]
       except Exception as e:
           print(f"Warning: developability scoring failed for seq_id={sid}: {e}")
           score = float("nan")

       df.at[idx, dev_col] = score

       # immediately persist after each new score
       df.to_json(interim_dev_file, orient="records", lines=True)

   print(f"Developability ready: {df[dev_col].notna().sum()} / {len(df)}")
   return df

def populate_training_metrics(
   train_jsonl: str,
   sim_jsonl: str,
   output_jsonl: str | None = None
) -> pd.DataFrame:
   """
   For sim_jsonl, fill in y_high and dev_score for the first 1024 sequences
   (training set) using values from train_jsonl.
  
   Parameters
   ----------
   train_jsonl : str
       Path to the “true” run JSONL (with y_high & dev_score present for seq_id 0–1023).
   sim_jsonl : str
       Path to the “simulated” run JSONL (where training rows have nulls).
   output_jsonl : str, optional
       If given, writes the merged DataFrame back out as JSONL.
  
   Returns
   -------
   pd.DataFrame
       The merged DataFrame (with no nulls in y_high/dev_score for seq_id < 1024).
   """
   # 1) Load both runs
   train_df = pd.read_json(train_jsonl, lines=True)
   sim_df   = pd.read_json(sim_jsonl,   lines=True)

   # 2) Build lookup maps for the training set
   lookup = train_df.set_index("seq_id")[["y_high", "dev_score"]]

   # 3) Identify the training rows in sim_df
   is_train = sim_df["seq_id"] < 1024

   # 4) Overwrite sim_df’s nulls with the true values
   sim_df.loc[is_train, "y_high"]    = sim_df.loc[is_train, "seq_id"].map(lookup["y_high"])
   sim_df.loc[is_train, "dev_score"] = sim_df.loc[is_train, "seq_id"].map(lookup["dev_score"])

   # 5) Filter: keep all training + only selected BO points
   train_rows = sim_df.loc[sim_df["seq_id"] < 1024]
   bo_rows = sim_df.loc[(sim_df["seq_id"] >= 1024) & (sim_df["selected"])]
   out_df = pd.concat([train_rows, bo_rows], ignore_index=True)
   out_df["pred_error"] = out_df["y_high"] - out_df["fitness"]

   # 6) Optional: write back out
   if output_jsonl:
       out_df.to_json(output_jsonl, orient="records", lines=True)

   return out_df