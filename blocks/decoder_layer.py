import torch
import torch.nn as nn
from layers.mha import MultiHeadAttention
from layers.layerNorm import LayerNorm
from layers.position_wise_feed_forward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(d_model)

        # In 3.1, decoder layer's part : the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack
        self.enc_dec_attention = MultiHeadAttention(d_model, n_head)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = LayerNorm(d_model)

    def forward(self, dec, enc, trg_mask, src_mask):
        # masked self-attn
        sa_in  = self.norm1(dec)
        sa_out = self.self_attention(sa_in, sa_in, sa_in, mask=trg_mask)
        x = dec + self.dropout1(sa_out)

        # cross-attn
        ca_in_q = self.norm2(x)
        ca_out  = self.enc_dec_attention(ca_in_q, enc, enc, mask=src_mask)
        x = x + self.dropout2(ca_out)

        # ffn
        ffn_in  = self.norm3(x)
        ffn_out = self.ffn(ffn_in)
        x = x + self.dropout3(ffn_out)
        return x
        # # 1) Masked Self-Attention (decoder-side causal mask + target padding)
        # # In the decoder's self-attention, we must prevent access to future tokens (causal masking).
        # # The trg_mask usually combines a causal mask (upper-triangular mask that blocks future positions)
        # # with a target padding mask (blocks padded positions in the target sequence).
        # _x = dec
        # x = self.self_attention(dec, dec, dec, mask=trg_mask)
        # x = self.norm1(_x + self.dropout1(x))

        # # 2) Encoder–Decoder (Cross) Attention (source padding only)
        # # Here, the query (Q) comes from the decoder output `x`, and the keys (K) and values (V)
        # # come from the encoder output `enc`.
        # # The src_mask is used only to block padding tokens in the source sequence.
        # # No causal masking is needed here because the source sequence is fully known.
        # if enc is not None:
        #     _x = x
        #     x = self.enc_dec_attention(x, enc, enc, mask=src_mask)   # <- Ensure consistent naming: enc_dec_attn
        #     x = self.norm2(_x + self.dropout2(x))

        # # 3) Position-wise Feed-Forward Network (position-independent MLP) + Post-Norm
        # # Applies the FFN to each position independently, then adds residual connection and LayerNorm.
        # _x = x
        # x = self.ffn(x)
        # x = self.norm3(_x + self.dropout3(x))
        # return x