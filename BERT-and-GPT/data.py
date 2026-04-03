import torch
import json
import os
from collections import Counter
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
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids):
        return "".join(self.itos.get(i, "") for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos, "vocab_size": self.vocab_size}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls.__new__(cls)
        vocab.stoi = data["stoi"]
        vocab.itos = {int(k): v for k, v in data["itos"].items()}
        vocab.vocab_size = data["vocab_size"]
        return vocab

def build_vocab_from_files(file_paths, min_freq=2, base_chars="abcdefghijklmnopqrstuvwxyz0123456789 .,!?\n"):
    if isinstance(file_paths, str): file_paths = [file_paths]
    counter = Counter()
    for fp in file_paths:
        with open(fp, "r", encoding="utf-8") as f:
            counter.update(f.read())
    valid_chars = set(base_chars) | {c for c, cnt in counter.items() if cnt >= min_freq}
    return SimpleVocab(chars="".join(sorted(valid_chars)))

def load_real_corpus(file_paths, vocab, seq_len=64):
    if isinstance(file_paths, str): file_paths = [file_paths]
    
    all_tokens = []
    for fp in file_paths:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"can't find: {fp}")
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()
        all_tokens.extend(vocab.encode(text))

    if len(all_tokens) < seq_len:
        raise ValueError(f"({len(all_tokens)}) < ({seq_len})")

    tensor_data = torch.tensor(all_tokens, dtype=torch.long)
    n_chunks = len(tensor_data) // seq_len
    tensor_data = tensor_data[:n_chunks * seq_len].view(-1, seq_len)
    return TensorDataset(tensor_data, tensor_data)