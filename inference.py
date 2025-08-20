import argparse
import torch
from model.transformer import Transformer
from data import load_translation, tokenize_de, Vocab  # re-use your tokenizers/Vocab

@torch.inference_mode()
def build_tgt_mask(ys, pad_id: int):
    L = ys.size(1)
    causal = torch.tril(torch.ones((1, 1, L, L), device=ys.device, dtype=torch.bool))
    pad_mask = (ys != pad_id).unsqueeze(1).unsqueeze(2)  # [1,1,1,L]
    return pad_mask & causal                              # [1,1,L,L]

@torch.inference_mode()
def greedy_decode(model, src_ids, src_mask, pad_id, bos_id, eos_id, max_len=256):
    device = src_ids.device
    ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)  # start with BOS
    for _ in range(max_len - 1):
        tgt_mask = build_tgt_mask(ys, pad_id)
        logits = model(src_ids, ys, src_mask, tgt_mask)             # [1, Lt, V]
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)     # [1,1]
        ys = torch.cat([ys, next_id], dim=1)
        if next_id.item() == eos_id:
            break
    return ys.squeeze(0)  # [<=max_len]

def ids_to_text(ids, vocab: Vocab, bos_id, eos_id):
    toks = []
    for i in ids.tolist():
        if i == bos_id:   # skip BOS
            continue
        if i == eos_id:   # stop at EOS
            break
        t = vocab.lookup_token(i)
        if t in ("<pad>", "<bos>", "<eos>"):
            continue
        toks.append(t)
    return " ".join(toks)

@torch.inference_mode()
def encode_src(text_de: str, src_vocab: Vocab, pad_id, bos_id, eos_id, max_src_len=256, device="cpu"):
    # IMPORTANT: your training used [BOS] + tokens + [EOS] on the source too
    toks = [tok.lower() for tok in tokenize_de(text_de)]
    toks = toks[: max_src_len - 2]                          # room for BOS/EOS
    ids = [bos_id] + [src_vocab[t] for t in toks] + [eos_id]
    src = torch.tensor([ids], dtype=torch.long, device=device)        # [1, Ls]
    src_mask = (src != pad_id).unsqueeze(1).unsqueeze(2).to(torch.bool)
    return src, src_mask

def rebuild_vocab_from_ckpt(itos_list):
    # reconstruct your simple Vocab from a saved itos list
    v = Vocab(counter={}, min_freq=1)  # dummy init
    v.itos = list(itos_list)
    v.stoi = {tok: i for i, tok in enumerate(v.itos)}
    v.default_index = v.stoi.get("<unk>", 3)
    return v

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    parser.add_argument("--src", type=str, required=True, help="German source sentence")
    parser.add_argument("--max_len", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)

    # --- vocab handling ---
    if "src_itos" in ckpt and "tgt_itos" in ckpt:
        src_vocab = rebuild_vocab_from_ckpt(ckpt["src_itos"])
        tgt_vocab = rebuild_vocab_from_ckpt(ckpt["tgt_itos"])
        SRC_VOC, TRG_VOC = len(src_vocab), len(tgt_vocab)
        PAD_ID, BOS_ID, EOS_ID = src_vocab["<pad>"], src_vocab["<bos>"], src_vocab["<eos>"]
    else:
        # fallback: rebuild via loader (must match training code)
        train_loader, _, _, sizes, ids, vocabs = load_translation(
            device=device, batch_size=1, min_freq=2, max_src_len=args.max_len, max_trg_len=args.max_len,
            num_workers=0, pin_memory=False
        )
        SRC_VOC, TRG_VOC = sizes
        PAD_ID, BOS_ID, EOS_ID = ids["pad"], ids["bos"], ids["eos"]
        src_vocab, tgt_vocab = vocabs["src"], vocabs["tgt"]

    # --- model ---
    model = Transformer(
        src_voc=SRC_VOC, trg_voc=TRG_VOC, max_len=args.max_len,
        d_model=512, ffn_hidden=2048, n_head=8, n_layers=6,
        dropout=0.1, device=device
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # --- encode + decode ---
    src_ids, src_mask = encode_src(args.src, src_vocab, pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID,
                                   max_src_len=args.max_len, device=device)
    out_ids = greedy_decode(model, src_ids, src_mask, PAD_ID, BOS_ID, EOS_ID, max_len=args.max_len)
    hypo = ids_to_text(out_ids, tgt_vocab, BOS_ID, EOS_ID)

    print("\n[DE]", args.src)
    print("[EN]", hypo)

if __name__ == "__main__":
    main()
