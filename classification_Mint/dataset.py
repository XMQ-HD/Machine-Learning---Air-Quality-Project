import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int, horizon: int = 1):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        self.horizon = horizon
        
        self.X_seq, self.y_seq = self._create_sequences()
        
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        X_seq, y_seq = [], []
        
        max_idx = len(self.X) - self.sequence_length - self.horizon + 1
        
        for i in range(max_idx):
            X_seq.append(self.X[i:i + self.sequence_length])
            y_seq.append(self.y[i + self.sequence_length + self.horizon - 1])
        
        return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.int64)
    
    def __len__(self):
        return len(self.X_seq)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X_seq[idx]), torch.LongTensor([self.y_seq[idx]])[0]