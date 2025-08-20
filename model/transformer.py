import torch.nn as nn
import torch
from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, src_voc, trg_voc, max_len, d_model=512, ffn_hidden=2048, n_head=8, n_layers=6, dropout=0.1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.device = device
        self.encoder = Encoder(src_voc, max_len, d_model, ffn_hidden, n_head, n_layers, dropout, device)
        self.decoder = Decoder(trg_voc, max_len, d_model, ffn_hidden, n_head, n_layers, dropout, device)
        self.generator = nn.Linear(d_model, trg_voc)

        self.apply(self._init_weights)

    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)
        
    def forward(self, src, trg, src_mask, trg_mask):
        enc_out = self.encoder(src, src_mask) # [B,L_s,D]
        dec_out = self.decoder(trg, enc_out, trg_mask, src_mask) # [B,L_t,D]
        logits = self.generator(dec_out) # [B, L_t, vocab]
        return logits
    
