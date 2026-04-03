import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_model(model, dataloader, epochs, lr, device, accum_steps=2, save_path="checkpoint.pt"):
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # dynamic learning rate, making training more efficient
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            batch = [x.to(device) for x in batch]
            input_ids = batch[0]

            # auto precision
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids)
                model_name = model.__class__.__name__
                if model_name == "MiniBERT":
                    labels, mask_labels = batch[1], batch[2]
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")
                    loss = (loss * mask_labels.view(-1)).sum() / (mask_labels.sum() + 1e-8)
                elif model_name == "MiniGPT":
                    targets = batch[1][:, 1:].contiguous()
                    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), targets.reshape(-1))
                else:
                    raise RuntimeError("Model Not Supported")
                loss = loss / accum_steps
            
            loss.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss.item() * accum_steps
            pbar.set_postfix({"loss": f"{loss.item()*accum_steps:.4f}"})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Finished. Avg Loss: {avg_loss:.4f}")
        torch.save({"model": model.state_dict(), "epoch": epoch+1, "lr": scheduler.get_last_lr()[0]}, save_path)
    
    print(f"Saved to {save_path}")