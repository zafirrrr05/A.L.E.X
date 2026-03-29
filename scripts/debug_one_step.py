import torch

from src.training.ssl_trainer import SSLTrainer


def main():

    torch.manual_seed(0)

    trainer = SSLTrainer(device="cpu")

    # -------------------------------------------------
    # fake batch for the "order" task
    # shapes must match what your SSLTrainer expects
    # -------------------------------------------------

    batch = {
        "task": "order",

        # clip length = 10
        "clip1_team1": torch.randn(10, 11, 4),
        "clip1_team2": torch.randn(10, 11, 4),
        "clip1_ball":  torch.randn(10, 4),

        "clip2_team1": torch.randn(10, 11, 4),
        "clip2_team2": torch.randn(10, 11, 4),
        "clip2_ball":  torch.randn(10, 4),

        "label": 1
    }

    loss = trainer.train_step(batch)

    print("Sanity check loss:", loss)


if __name__ == "__main__":
    main()