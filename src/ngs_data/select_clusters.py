from __future__ import annotations
from typing import Union, Optional
import numpy as np
import pandas as pd
import hdbscan
from pathlib import Path
from src.embeddings.esmc_encoder import embed_sequences


def select_sequences(
    input_json: Union[str, Path],
    seq_col: str = "sequence",
    enrichment_col: str = "enrichment",
    embed_components: int = 32,
    n_select: int = 1200,
    min_cluster_size: int = 1000,
    save_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Perform NGS clustering and selection:

    1. Read a newline-delimited JSON with at least the sequence and enrichment columns.
    2. Generate ESM-C embeddings and PCA-reduced vectors.
    3. Weight each row by sqrt(enrichment) and expand the dataset accordingly.
    4. Cluster expanded embeddings with HDBSCAN.
    5. Select top sequences per cluster up to `n_select` total.
    6. Return a DataFrame with original data plus `cluster` and `r1_test` flags.

    Parameters
    ----------
    input_json : str or Path
        Path to input JSON (records, one per line).
    seq_col : str
        Column name containing sequences.
    enrichment_col : str
        Column name for enrichment scores.
    embed_components : int
        Number of PCA components for embeddings.
    n_select : int
        Total number of sequences to select.
    min_cluster_size : int
        HDBSCAN `min_cluster_size` parameter.
    save_path : str or Path, optional
        If provided, save selected DataFrame to this path as JSONL.

    Returns
    -------
    pd.DataFrame
        Selected subset with added `cluster` and `r1_test` columns.
    """
    # Load input
    df = pd.read_json(input_json, lines=True)

    # 1) Embed sequences via ESM-C + PCA
    df = embed_sequences(df, seq_col=seq_col, n_components=embed_components)
    emb_col = f"embedding_pca{embed_components}"
    emb = np.vstack(df[emb_col].values)

    # 2) Weight by sqrt(enrichment) and expand
    reps = np.floor(np.sqrt(df[enrichment_col])).clip(min=1).astype(int)
    df_exp = df.loc[df.index.repeat(reps)].reset_index(drop=True)
    emb_exp = np.repeat(emb, reps, axis=0)

    # 3) Cluster with HDBSCAN
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(emb_exp)
    df_exp["cluster"] = labels
    clusters = [c for c in np.unique(labels) if c >= 0] or [0]
    per_cluster = int(np.ceil(n_select / len(clusters)))

    # 4) Select top sequences per cluster
    selected = []
    for c in clusters:
        sub = df_exp[df_exp.cluster == c]
        top = (
            sub.sort_values(enrichment_col, ascending=False)
               .drop_duplicates(seq_col)
               .head(per_cluster)
        )
        selected.extend(top[seq_col].tolist())

    # 5) Deduplicate & fill up to n_select
    seen = {}
    unique_selected = []
    for s in selected:
        if s not in seen:
            seen[s] = True
            unique_selected.append(s)
        if len(unique_selected) >= n_select:
            break
    if len(unique_selected) < n_select:
        noise = df_exp[df_exp.cluster == -1]
        for s in noise.sort_values(enrichment_col, ascending=False)[seq_col]:
            if s not in seen:
                seen[s] = True
                unique_selected.append(s)
            if len(unique_selected) >= n_select:
                break

    # 6) Build final DataFrame
    final = df[df[seq_col].isin(unique_selected)].copy()
    cluster_map = df_exp.drop_duplicates(seq_col).set_index(seq_col).cluster
    final["cluster"] = final[seq_col].map(cluster_map)
    final["r1_test"] = True

    # Save if requested
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        final.to_json(save_path, orient="records", lines=True)

    return final
