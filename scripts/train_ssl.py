import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.ssl_dataset import TeamSequenceDataset
from src.training.ssl_collate import SSLTaskSampler
from src.training.ssl_trainer import SSLTrainer

# python -m scripts.train_ssl
def main():

    ds = TeamSequenceDataset("data/sequences", min_len=40)

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        collate_fn=SSLTaskSampler(),
        num_workers=0,   # <- important on Windows
        pin_memory=False
    )


    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"[Detector] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[Detector] Using CPU") 


    trainer = SSLTrainer(device=device)
    pbar = tqdm(enumerate(loader), total=5000, ncols=80)

    for step, batch in pbar:
        loss = trainer.train_step(batch)

        # progress bar always updates
        pbar.set_description(f"step {step:>4} | {batch['task']:<12} | loss {loss:.4f}")

        # clean log line every 100 steps
        if step % 100 == 0:
            tqdm.write(f"step {step:>4} | task {batch['task']:<12} | loss {loss:.4f}")

        if step >= 5000:
            break

    # ── checkpoint to load into the GATv2 later ──
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(trainer.model.state_dict(), "checkpoints/ssl_encoder.pt")
    print("Saved: checkpoints/ssl_encoder.pt")    


if __name__ == "__main__":
    main()