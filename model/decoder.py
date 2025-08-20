import torch
import torch.nn as nn
from embedding.transformer_embedding import TransformerEmbedding
from blocks.decoder_layer import DecoderLayer

class Decoder(nn.Module):
    def __init__(self, dec_vocab, max_len, d_model, ffn_hidden, n_head, n_layers, dropout, device):
        super().__init__()
        self.emb = TransformerEmbedding(dec_vocab, d_model, max_len, dropout, device)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, ffn_hidden, n_head, dropout) for _ in range(n_layers)
        ])

    def forward(self, tgt, enc_out, trg_mask, src_mask):
        x = self.emb(tgt)
        for layer in self.layers:
            x = layer(x, enc_out, trg_mask, src_mask)
        return x