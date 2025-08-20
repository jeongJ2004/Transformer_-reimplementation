import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import spacy
from collections import Counter

# ---- Load Spacy Tokenizers ----
spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")

def tokenize_de(text):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

# ---- Vocab Implementation ----
class Vocab:
    def __init__(self, counter, specials=["<pad>", "<bos>", "<eos>", "<unk>"], min_freq=2):
        self.itos = list(specials)
        for token, freq in counter.items():
            if freq >= min_freq and token not in specials:
                self.itos.append(token)
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}
        self.default_index = self.stoi["<unk>"]

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi.get(token, self.default_index)

    def lookup_token(self, idx):
        return self.itos[idx]

def build_vocab(data_iter, lang, min_freq=2):
    counter = Counter()
    for sample in data_iter:
        if lang == "de":
            counter.update(tokenize_de(sample["translation"]["de"]))
        else:
            counter.update(tokenize_en(sample["translation"]["en"]))
    return Vocab(counter, min_freq=min_freq)

# ---- Collate (CPU tensors only) ----
def collate_batch(batch, src_vocab, trg_vocab, pad_id, bos_id, eos_id,
                  max_src_len, max_trg_len):
    src_list, trg_in_list, trg_out_list = [], [], []

    for sample in batch:
        src_tokens = [bos_id] + [src_vocab[token] for token in tokenize_de(sample["translation"]["de"])] + [eos_id]
        trg_tokens = [bos_id] + [trg_vocab[token] for token in tokenize_en(sample["translation"]["en"])] + [eos_id]

        src_tokens = src_tokens[:max_src_len]
        trg_tokens = trg_tokens[:max_trg_len]

        src = torch.tensor(src_tokens, dtype=torch.long)
        trg_in = torch.tensor(trg_tokens[:-1], dtype=torch.long)
        trg_out = torch.tensor(trg_tokens[1:], dtype=torch.long)

        src_list.append(src)
        trg_in_list.append(trg_in)
        trg_out_list.append(trg_out)

    src_batch = pad_sequence(src_list, padding_value=pad_id, batch_first=True)
    trg_in_batch = pad_sequence(trg_in_list, padding_value=pad_id, batch_first=True)
    trg_out_batch = pad_sequence(trg_out_list, padding_value=pad_id, batch_first=True)

    # Return masks as None; training loop will build them on-device
    return src_batch, trg_in_batch, trg_out_batch, None, None

# ---- Loader ----
def load_translation(
    device, batch_size=64, min_freq=2,
    max_src_len=256, max_trg_len=256,
    num_workers=2, pin_memory=True
):
    # Single split dataset; we split 90/5/5
    ds = load_dataset("opus_books", "de-en")
    full_train = ds["train"]

    split = full_train.train_test_split(test_size=0.1, seed=42)
    train_data = split["train"]
    temp = split["test"].train_test_split(test_size=0.5, seed=42)
    valid_data = temp["train"]
    test_data = temp["test"]

    # Build vocabs on train only
    src_vocab = build_vocab(train_data, "de", min_freq=min_freq)
    trg_vocab = build_vocab(train_data, "en", min_freq=min_freq)

    PAD_ID, BOS_ID, EOS_ID = src_vocab["<pad>"], src_vocab["<bos>"], src_vocab["<eos>"]
    ids = {"pad": PAD_ID, "bos": BOS_ID, "eos": EOS_ID}
    sizes = (len(src_vocab), len(trg_vocab))
    vocabs = {"src": src_vocab, "tgt": trg_vocab}

    # NOTE: do NOT pass device here
    def _collate(batch):
        return collate_batch(batch, src_vocab, trg_vocab, PAD_ID, BOS_ID, EOS_ID, max_src_len, max_trg_len)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              collate_fn=_collate, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False,
                              collate_fn=_collate, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False,
                              collate_fn=_collate, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader, sizes, ids, vocabs
