import torch
import torch.nn as nn 
from layers.scaledotproduct import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, attn_drop=0.1, proj_drop=0.0):
        super().__init__()
        assert d_model % n_head == 0

        self.n_head = n_head
        self.d_head = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj_dropout = nn.Dropout(proj_drop)

    def split(self, x):
        # x: [B, L, D] -> [B, H, L, d_head], B : batch size, L : sequence length (# of tokens), D = model/token dimension == d_model here
        B, L, D = x.size()
        # (B, L, H, d_head) -> (B, H, L, d_head)
        return x.view(B, L, self.n_head, self.d_head).transpose(1, 2)
    
    def concat(self, x):
        # We want [B, H, L, d_head] -> [B, L, D]
        B, H, L, d_head = x.size()
        return x.transpose(1, 2).contiguous().view(B, L, H * d_head) # cuz D = H * d_head
    
    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, attn = self.attention(q, k, v, mask=mask)
        attn = self.attn_dropout(attn)
        out = attn @ v
        out = self.concat(out)
        out = self.w_o(out)
        out = self.proj_dropout(out)
        return out
        # q, k, v = self.w_q(q), self.w_k(k), self.w_v(v) # Q = XW^Q, K = XW^K ...
        # q, k, v = self.split(q), self.split(k), self.split(v)

        # out, attn = self.attention(q, k, v, mask=mask)

        # out = self.concat(out)

        # out = self.w_o(out)

        # return out
    

# One general remark that I've learned : instead of doinf split, concat functions, I could simply use "einops" module in order to simplify these tasks.
# For this version, I'll manually implement split and concat funcs but for the future one, I should learn this einops funcs.
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py check here, from line 50
