import torch
from torch.utils.data import TensorDataset

class SimpleVocab:
    def __init__(self, chars="abcdefghijklmnopqrstuvwxyz0123456789 .,!?\n"):
        self.stoi = {c: i + 1 for i, c in enumerate(chars)}
        self.stoi["[MASK]"] = len(self.stoi) + 1
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(self.stoi) + 1

    @property
    def pad_id(self): return 0
    @property
    def mask_id(self): return self.vocab_size - 1

    def encode(self, text):
        return [self.stoi.get(c, 0) for c in text.lower()]

    def decode(self, ids):
        return "".join(self.itos.get(i, "") for i in ids)

def build_dataset(vocab, n_samples=2000, seq_len=64, mode="gpt"):
    all_ids = torch.randint(1, vocab.vocab_size - 1, (n_samples, seq_len), dtype=torch.long)

    if mode == "bert":
        pass
    elif mode == "gpt":
        return TensorDataset(all_ids, all_ids)
    else:
        raise RuntimeError(f"mode {mode} is not supported!")