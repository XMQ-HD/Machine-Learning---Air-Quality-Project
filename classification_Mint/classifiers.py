import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple, Dict

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

from dataset import TimeSeriesDataset
from models import RNNClassifierModel


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state = None
        self.should_stop = False
        
    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best validation loss: {self.best_loss:.4f}")
                self.should_stop = True
                return True
            
            return False
    
    def restore_best_model(self, model: nn.Module):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class RNNClassifier:
    
    def __init__(
        self,
        model_type: str = 'lstm',
        sequence_length: int = 24,
        hidden_units: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        device: str = None
    ):
        
        self.model_type = model_type.lower()
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.history = None
        
    def build_model(self, input_dim: int, num_classes: int = 3):
        self.model = RNNClassifierModel(
            input_dim=input_dim,
            hidden_units=self.hidden_units,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            model_type=self.model_type,
            num_classes=num_classes
        ).to(self.device)
        
        return self.model
    
    def _process_single_batch(
        self, 
        batch_X: torch.Tensor, 
        batch_y: torch.Tensor,
        criterion: nn.Module,
        optimizer: optim.Optimizer = None,
        is_training: bool = True
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        batch_X = batch_X.to(self.device)
        batch_y = batch_y.to(self.device)
        
        if is_training:
            optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
        
        _, predicted = torch.max(outputs.data, 1)
        
        return loss.item(), predicted.cpu().numpy(), batch_y.cpu().numpy()
    
    def _run_epoch(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer = None,
        is_training: bool = True
    ) -> Tuple[float, float, float, float]:
        if is_training:
            self.model.train()
        else:
            self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        batch_iterator = iter(data_loader)
        batch_idx = 0
        
        while batch_idx < len(data_loader):
            try:
                batch_X, batch_y = next(batch_iterator)
                
                loss, preds, labels = self._process_single_batch(
                    batch_X, batch_y, criterion, optimizer, is_training
                )
                
                total_loss += loss
                all_preds.extend(preds)
                all_labels.extend(labels)
                
                batch_idx += 1
            except StopIteration:
                break
        
        # matrics
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_loss, accuracy, precision, recall
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        horizon: int = 1,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10
    ):
        train_dataset = TimeSeriesDataset(X_train, y_train, self.sequence_length, horizon)
        val_dataset = TimeSeriesDataset(X_val, y_val, self.sequence_length, horizon)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"\n{'='*60}")
        print(f"Training {self.model_type.upper()} Model - Forecasting CO {horizon}h ahead")
        print(f"{'='*60}")
        
        if self.model is None:
            self.build_model(input_dim=train_dataset.X_seq.shape[2])
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        self.history = {
            'loss': [], 'val_loss': [],
            'accuracy': [], 'val_accuracy': [],
            'precision': [], 'val_precision': [],
            'recall': [], 'val_recall': []
        }

        early_stopping = EarlyStopping(patience=patience)

        epoch = 0
        training_active = True
        
        while epoch < epochs and training_active:
            train_loss, train_acc, train_prec, train_rec = self._run_epoch(
                train_loader, criterion, optimizer, is_training=True
            )
            
            val_loss, val_acc, val_prec, val_rec = self._run_epoch(
                val_loader, criterion, is_training=False
            )
            
            # history
            self.history['loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['accuracy'].append(train_acc)
            self.history['val_accuracy'].append(val_acc)
            self.history['precision'].append(train_prec)
            self.history['val_precision'].append(val_prec)
            self.history['recall'].append(train_rec)
            self.history['val_recall'].append(val_rec)

            scheduler.step(val_loss)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # early stopping
            if early_stopping(val_loss, self.model, epoch):
                early_stopping.restore_best_model(self.model)
                training_active = False
            
            epoch += 1
        
        return self.history
    
    def predict(
        self,
        X: np.ndarray,
        y: np.ndarray,
        horizon: int = 1,
        batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        dataset = TimeSeriesDataset(X, y, self.sequence_length, horizon)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        y_pred = []
        y_true = []
        
        batch_iterator = iter(loader)
        batch_idx = 0
        
        with torch.no_grad():
            while batch_idx < len(loader):
                try:
                    batch_X, batch_y = next(batch_iterator)
                    batch_X = batch_X.to(self.device)
                    
                    outputs = self.model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    y_pred.extend(predicted.cpu().numpy())
                    y_true.extend(batch_y.cpu().numpy())
                    
                    batch_idx += 1
                except StopIteration:
                    break
        
        return np.array(y_true), np.array(y_pred)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        horizon: int = 1
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        y_true, y_pred = self.predict(X, y, horizon)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics, y_true, y_pred
    
    def save_model(self, path: Path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'sequence_length': self.sequence_length,
            'hidden_units': self.hidden_units,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate
        }, path)
    
    def load_model(self, path: Path, input_dim: int, num_classes: int = 3):
        checkpoint = torch.load(path)
        
        self.model_type = checkpoint['model_type']
        self.sequence_length = checkpoint['sequence_length']
        self.hidden_units = checkpoint['hidden_units']
        self.num_layers = checkpoint['num_layers']
        self.dropout_rate = checkpoint['dropout_rate']
        self.learning_rate = checkpoint['learning_rate']
        
        self.build_model(input_dim, num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()


class NaiveClassifier:
    def predict(
        self,
        y: np.ndarray,
        horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:

        y_true = y[horizon:]
        y_pred = y[:-horizon]
        return y_true, y_pred
    
    def evaluate(
        self,
        y: np.ndarray,
        horizon: int = 1
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:

        y_true, y_pred = self.predict(y, horizon)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics, y_true, y_pred