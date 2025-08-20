import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout): # d_ff = 2048, d_model = 512, dropout rate = 0.1 are specified in the paper
        super().__init__()
        # FFN(x) = max(0, xW1 + b1) W2 + b2 
        self.lin1 = nn.Linear(d_model, hidden)
        self.lin2 = nn.Linear(hidden, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.dropout(F.relu(self.lin1(x)))) 