import torch

def gpt(model, prompt_ids, vocab, max_new=30, temperature=0.8, device="cuda"):
    model.eval()
    x = prompt_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(max_new):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            logits = logits[:, -1, :] / temperature # temp controls the shape of distribution
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_id], dim=1)

            # end of the conversation
            if next_id.item() == vocab.pad_id:
                break 
    return vocab.decode(x.squeeze().cpu().tolist())