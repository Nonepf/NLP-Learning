"""
BERT not Implemented
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from models import MiniGPT
from data import SimpleVocab, build_dataset
from train import train_model
from inference import gpt

def parseArgs():
    parser = argparse.ArgumentParser(description="language models")
    parser.add_argument("--mode", choices=["train", "interact"], default="train")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--ckpt", type=str, default="checkpoint.pt")
    return parser.parse_args()

def runTraining(vocab, model, device, args):
    ds = build_dataset(vocab, n_samples=3000, seq_len=64, mode="gpt")
    loader = DataLoader(ds, batch_size=64, shuffle=True, drop_last=False)
    print("Start to train GPT...")
    train_model(model, loader, epochs=args.epochs, lr=3e-4, device=device, save_path=args.ckpt)

def runInference(vocab, model, device, args):
    assert os.path.exists(args.ckpt), f"no checkpoint: {args.ckpt}"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print("Model Loaded.")
    
    while True:
        prompt = input("\nYou: ")
        if prompt.lower() == "quit": break
        ids = torch.tensor(vocab.encode(prompt), dtype=torch.long)
        print("GPT:", gpt(model, ids, vocab, max_new=40))

def main():
    args = parseArgs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    vocab = SimpleVocab()
    model = MiniGPT(vocab.vocab_size, d_model=256, n_layers=4, max_len=64)

    if args.mode == "train":
        runTraining(vocab, model, device, args)
    elif args.mode == "interact":
        runInference(vocab, model, device, args)

if __name__ == "__main__":
    main()