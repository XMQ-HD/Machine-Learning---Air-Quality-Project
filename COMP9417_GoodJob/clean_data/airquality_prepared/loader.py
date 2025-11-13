import json
from pathlib import Path
import pandas as pd

def load_pack(base_dir: str, pack='trees', split='train'):
    base = Path(base_dir)
    X = pd.read_parquet(base/'features'/pack/'X.parquet')
    y = pd.read_parquet(base/'features'/pack/'y.parquet')
    splits = json.loads((base/'splits.json').read_text())
    idx = splits[split]
    return (X.iloc[idx].reset_index(drop=True),
            y['target_reg'].iloc[idx].reset_index(drop=True),
            y['target_cls'].iloc[idx].reset_index(drop=True))
