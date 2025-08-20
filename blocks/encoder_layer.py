import torch
import torch.nn as nn 
from layers.mha import MultiHeadAttention
from layers.layerNorm import LayerNorm
from layers.position_wise_feed_forward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, dropout): # in the paper, d_model = 512, d_ff = 2048, h = n_head = 8, dropout rate = 0.1
        super().__init__()
        # Gotta read 3.1 of the paper for encoder layer part 
        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(d_model) # LayerNorm(x + Dropout(SelfAttn(x))) == LayerNorm(x + Sublayer(x))
        
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, src_mask):
        sa_in = self.norm1(x)
        sa_out = self.self_attention(sa_in, sa_in, sa_in, mask=src_mask)
        x = x + self.dropout1(sa_out)

        ffn_in = self.norm2(x)
        ffn_out = self.ffn(ffn_in)
        x = x + self.dropout2(ffn_out)
        return x
        # _x = x # Copying the original x for the "Residual" part
        # x = self.self_attention(x, x, x, mask=src_mask) # q=k=v hence we all put x
        # x = self.norm1(_x + self.dropout1(x)) # check 5.3 : the output of each sub-layer is LayerNorm(x + Sublayer(x))” + “We apply dropout to the output of each sub-layer, before it is added… and normalized
        # _x = x
        # x = self.ffn(x)
        # x = self.norm2(_x + self.dropout2(x))
        # return x
