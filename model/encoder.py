import torch
import torch.nn as nn 
from embedding.transformer_embedding import TransformerEmbedding
from blocks.encoder_layer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, enc_vocab, max_len, d_model, ffn_hidden, n_head, n_layers, dropout, device): # enc_vocab = input vocab size, ffn_hidden == d_ff, n_layers = 6
        super().__init__()
        self.emb = TransformerEmbedding(enc_vocab, d_model, max_len, dropout, device)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, ffn_hidden, n_head, dropout) for _ in range(n_layers)
        ])

    def forward(self, x, src_mask):
        x = self.emb(x) # [B, L_s, D]
        for layer in self.layers:
            x = layer(x, src_mask) # EncodeLayer: Self-attention -> FFN 
        return x # enc_out (mem) = [B, L_s, D]