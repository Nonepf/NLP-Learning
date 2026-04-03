import argparse
import os
import torch
from torch.utils.data import DataLoader
from models import MiniGPT
from data import SimpleVocab, build_vocab_from_files, load_real_corpus
from train import train_model
from inference import gpt

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--mode", choices=["train", "interact"], default="train")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--ckpt", type=str, default="checkpoint.pt")
    parser.add_argument("--data", nargs="+", default=["shakespeare.txt"])
    parser.add_argument("--vocab", type=str, default="vocab.json")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2048)
    return parser.parse_args()

def run_training(vocab, model, device, args):
    ds = load_real_corpus(args.data, vocab, seq_len=args.seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    print(f"Start Training GPT | sample: {len(ds)} | vocab: {vocab.vocab_size}")
    train_model(model, loader, epochs=args.epochs, lr=3e-4, device=device, save_path=args.ckpt)
    vocab.save(args.vocab)
    print(f"vocab saved to: {args.vocab}")

def run_inference(vocab, model, device, args):
    assert os.path.exists(args.ckpt), f"can't find checkpoint: {args.ckpt}"
    assert os.path.exists(args.vocab), f"can't find vocab: {args.vocab}"
    
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print("Model Loaded.")

    while True:
        prompt = input("\nYou: ")
        if prompt.lower() == "quit": break
        ids = torch.tensor(vocab.encode(prompt), dtype=torch.long)
        print("GPT:", gpt(model, ids, vocab, max_new=50))

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.mode == "train":
        vocab = build_vocab_from_files(args.data)
        model = MiniGPT(vocab.vocab_size, d_model=256, n_layers=4, max_len=args.seq_len)
        run_training(vocab, model, device, args)
    elif args.mode == "interact":
        vocab = SimpleVocab.load(args.vocab)
        model = MiniGPT(vocab.vocab_size, d_model=256, n_layers=4, max_len=args.seq_len)
        run_inference(vocab, model, device, args)

if __name__ == "__main__":
    main()