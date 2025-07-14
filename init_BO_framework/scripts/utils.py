import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

def load_data(path: str, features: List[str], target: str = 'y') -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load JSON (newline-delimited) or CSV, return raw X, y arrays and full DataFrame.
    """
    if path.endswith('.json'):
        df = pd.read_json(path, orient='records', lines=True)
    else:
        df = pd.read_csv(path)
    x_raw = df[features].values
    y = df[target].values
    return x_raw, y, df

def preprocess(x_raw: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw)
    return x_scaled, scaler
