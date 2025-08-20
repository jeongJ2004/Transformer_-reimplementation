import torch 
import torch.nn as nn 
from embedding.positional_encoding import PositionalEncoding

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, dropout: float, device: torch.device, padding_idx = 0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx) # Basically, vocab_size == totla number of tokens (column of the emb table)
        self.pos_enc = PositionalEncoding(d_model, max_len, device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x : [B, L]
        pe = self.pos_enc(x)                    # [L, D]
        # this part is the part where we show why pos enc is important. Since we don't know the order of the words, we add the PE to the tok_emb
        x = self.tok_emb(x) + pe.unsqueeze(0)   # [B, L, D], unsqueeze(0) makes [L, D] -> [1, L, D] and tok_emb is [B, L, D]
        return self.dropout(x)
    
    