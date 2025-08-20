import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import load_translation
from model.transformer import Transformer

# ---- mask utils ----
def make_src_mask(src, pad_id: int):
    # [B, L] -> [B,1,1,L]
    return (src != pad_id).unsqueeze(1).unsqueeze(2)

def make_trg_mask(ti, pad_id: int):
    # padding + causal
    pad_mask = (ti != pad_id).unsqueeze(1).unsqueeze(2)  # [B,1,1,Lt]
    L = ti.size(1)
    causal = torch.tril(torch.ones((1,1,L,L), device=ti.device, dtype=torch.bool))
    return pad_mask & causal

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, ignore_idx=0):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.ignore_idx = ignore_idx

    def forward(self, pred, target):
        pred = pred.view(-1, pred.size(-1))
        target = target.view(-1)

        ignore = target == self.ignore_idx
        t = target.clone()
        t[ignore] = 0

        with torch.no_grad():
            dist = torch.full_like(pred, self.smoothing / (self.classes - 1))
            dist.scatter_(1, t.unsqueeze(1), self.confidence)
            dist[ignore] = 0.0

        logp = torch.log_softmax(pred, dim=-1)
        loss = -(dist * logp).sum(dim=1)
        return loss.masked_select(~ignore).mean()

class NoamOpt:
    def __init__(self, d_model, warmup, optimizer):
        self.optimizer = optimizer
        self._s = 0
        self.warmup = warmup
        self.factor = d_model ** (-0.5)

    def step(self):
        self._s += 1
        lr = self.factor * min(self._s ** (-0.5), self._s * (self.warmup ** (-1.5)))
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

def train_epoch(model, loader, optimizer, criterion, pad_id, device, amp=True, grad_clip=1.0):
    model.train()
    scaler = torch.amp.GradScaler('cuda', enabled=amp and torch.cuda.is_available())
    running, seen = 0.0, 0

    pbar = tqdm(loader, desc="train", leave=False)
    for src, ti, to, _, _ in pbar:
        src = src.to(device, non_blocking=True)
        ti  = ti.to(device, non_blocking=True)
        to  = to.to(device, non_blocking=True)

        src_mask = make_src_mask(src, pad_id)
        trg_mask = make_trg_mask(ti,  pad_id)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda', enabled=amp and torch.cuda.is_available()):
            logits = model(src, ti, src_mask, trg_mask)
            loss = criterion(logits, to)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer.optimizer)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer.optimizer)
        scaler.update()
        optimizer.step()

        bs = src.size(0)
        running += loss.item() * bs
        seen += bs
        pbar.set_postfix(loss=f"{running/max(seen,1):.4f}")

    return running / max(seen, 1)

@torch.no_grad()
def evaluate(model, loader, criterion, pad_id, device, amp=False):
    model.eval()
    running, seen = 0.0, 0
    pbar = tqdm(loader, desc="valid", leave=False)
    for src, ti, to, _, _ in pbar:
        src = src.to(device, non_blocking=True)
        ti  = ti.to(device, non_blocking=True)
        to  = to.to(device, non_blocking=True)

        src_mask = make_src_mask(src, pad_id)
        tgt_mask = make_trg_mask(ti,  pad_id)

        with torch.amp.autocast(device_type='cuda', enabled=amp and torch.cuda.is_available()):
            logits = model(src, ti, src_mask, tgt_mask)
            loss = criterion(logits, to)

        bs = src.size(0)
        running += loss.item() * bs
        seen += bs
        pbar.set_postfix(loss=f"{running/max(seen,1):.4f}")
    return running / max(seen, 1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data ----
    train_loader, valid_loader, test_loader, sizes, ids, vocabs = load_translation(
        device=device, batch_size=64, min_freq=2, max_src_len=256, max_trg_len=256,
        num_workers=2, pin_memory=True
    )
    SRC_VOC, TRG_VOC = sizes
    PAD_ID = ids["pad"]; BOS_ID = ids["bos"]; EOS_ID = ids["eos"]

    # ---- Model ----
    model = Transformer(
        src_voc=SRC_VOC, trg_voc=TRG_VOC, max_len=256,
        d_model=512, ffn_hidden=2048, n_head=8, n_layers=6,
        dropout=0.1, device=device
    ).to(device)

    # ---- Loss/Opt ----
    criterion = LabelSmoothingLoss(classes=TRG_VOC, smoothing=0.1, ignore_idx=PAD_ID)
    base_optim = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    optimizer = NoamOpt(d_model=512, warmup=4000, optimizer=base_optim)

    # ---- Train ----
    EPOCHS = 10
    best = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    for ep in range(1, EPOCHS + 1):
        tr = train_epoch(model, train_loader, optimizer, criterion, PAD_ID, device, amp=True, grad_clip=1.0)
        va = evaluate(model, valid_loader, criterion, PAD_ID, device)
        lr = base_optim.param_groups[0]["lr"]
        print(f"[{ep:02d}] train={tr:.4f} valid={va:.4f} lr={lr:.6f}")

        if va < best:
            best = va
            torch.save({
                "model": model.state_dict(),
                "optim": base_optim.state_dict(),
                "epoch": ep,
                "pad_id": PAD_ID, "bos_id": BOS_ID, "eos_id": EOS_ID,
                "src_vocab_size": SRC_VOC, "tgt_vocab_size": TRG_VOC
            }, "checkpoints/best.pt")

    torch.save({
        "model": model.state_dict(),
        "optim": base_optim.state_dict(),
        "epoch": EPOCHS,
        "pad_id": PAD_ID, "bos_id": BOS_ID, "eos_id": EOS_ID,
        "src_vocab_size": SRC_VOC, "tgt_vocab_size": TRG_VOC
    }, "checkpoints/last.pt")

if __name__ == "__main__":
    main()
