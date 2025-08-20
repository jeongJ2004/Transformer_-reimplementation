import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, device:  torch.device): # Matrix -> max_len X d_model size 
        super().__init__()
        pe = torch.zeros(max_len, d_model, device=device)
        pe.requires_grad = False        

        position = torch.arange(0, max_len, device=device).float().unsqueeze(1) # [max_len, 1]
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        pe[:, 0::2] = torch.sin(position / (10000 ** (_2i / d_model)))
        pe[:, 1::2] = torch.cos(position / (10000 ** (_2i / d_model)))
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            seq_len = x.size(1)
        else:
            seq_len = x.size(1)
        return self.pe[:seq_len, :]



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pe = PositionalEncoding(d_model=8, max_len=10, device=device) 
