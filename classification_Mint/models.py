import torch.nn as nn


class RNNClassifierModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        model_type: str = 'lstm',
        num_classes: int = 3
    ):
        super(RNNClassifierModel, self).__init__()
        
        self.model_type = model_type.lower()
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        if self.model_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_units,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_units,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0
            )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_units, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        if self.model_type == 'lstm':
            out, (hidden, cell) = self.rnn(x)
        else:
            out, hidden = self.rnn(x)

        out = out[:, -1, :]
        
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out