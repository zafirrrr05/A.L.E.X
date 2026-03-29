import random
import torch
from .ssl_dataset import (
    sample_future_prediction,
    sample_masked_players,
    sample_possession,
    sample_temporal_order,
    sample_contrastive
)


TASKS = [
    "future",
    "future",       # doubled weight
    "masked",
    "possession",
    "possession",   # doubled weight
    "order",
    "contrastive"   # kept once — sampled ~14% instead of 20%
]


class SSLTaskSampler:

    def __init__(self, task_probs=None):
        if task_probs is None:
            self.task_probs = None
        else:
            self.task_probs = task_probs

    def __call__(self, batch):

        seq = batch[0]

        task = random.choice(TASKS)

        if task == "future":
            return {"task": "future", **sample_future_prediction(seq)}

        if task == "masked":
            return {"task": "masked", **sample_masked_players(seq)}

        if task == "possession":
            return {"task": "possession", **sample_possession(seq)}

        if task == "order":
            c1_t1, c1_t2, c2_t1, c2_t2, c1_b, c2_b, y = sample_temporal_order(seq)

            return {
                "task": "order",
                "clip1_team1": c1_t1,
                "clip1_team2": c1_t2,
                "clip2_team1": c2_t1,
                "clip2_team2": c2_t2,
                "clip1_ball":  c1_b,
                "clip2_ball":  c2_b,
                "label": torch.tensor(y, dtype=torch.long)
            }

        if task == "contrastive":
            return {"task": "contrastive", **sample_contrastive(seq)}
            
        raise ValueError(f"Unknown SSL task: {task}")
        
        