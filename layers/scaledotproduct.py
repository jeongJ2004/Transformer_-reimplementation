import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """
    q, k, v: [B, H, Lq, d_k], [B, H, Lk, d_k], [B, H, Lk, d_k]
    mask:    broadcastable to [B, H, Lq, Lk]; True = keep, False = mask
    """
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [B,H,Lq,Lk]

        if mask is not None:
            # ensure boolean mask where True=keep
            if mask.dtype != torch.bool:
                mask = mask != 0
            # masked softmax (stable): zero out masked positions before normalization
            # subtract max for stability
            # set masked scores to very negative by multiplying with mask after exp
            max_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min).max(dim=-1, keepdim=True).values
            scores = scores - max_scores
            exp_scores = torch.exp(scores) * mask.to(scores.dtype)
            denom = exp_scores.sum(dim=-1, keepdim=True) + 1e-9
            attn = exp_scores / denom
        else:
            attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)  # [B,H,Lq,d_k]
        return out, attn
