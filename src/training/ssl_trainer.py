import torch
import torch.nn as nn

from src.models.team_tactical_net import TacticalModel


class SSLTrainer:

    def __init__(self, device="cuda"):

        self.device = device

        self.model = TacticalModel(dim=64).to(device)

        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4
        )

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    # ------------------------------------------------------
    # utilities
    # ------------------------------------------------------

    def forward_backbone(self, team1, team2, ball):
        return self.model.forward_backbone(team1, team2, ball)

    # ------------------------------------------------------
    # training step
    # ------------------------------------------------------

    def train_step(self, batch):

        task = batch["task"]

        self.model.train()
        self.optim.zero_grad()

        if task == "future":
            loss = self._step_future(batch)

        elif task == "masked":
            loss = self._step_masked(batch)

        elif task == "possession":
            loss = self._step_possession(batch)

        elif task == "order":
            loss = self._step_order(batch)

        elif task == "contrastive":
            loss = self._step_contrastive(batch)

        else:
            raise ValueError(task)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optim.step()

        return loss.item()

    # ------------------------------------------------------
    # task implementations
    # ------------------------------------------------------

    # 1) future state prediction
    def _step_future(self, batch):

        t1 = batch["past_team1"].unsqueeze(0).to(self.device)
        t2 = batch["past_team2"].unsqueeze(0).to(self.device)
        b  = batch["past_ball"].unsqueeze(0).to(self.device)

        target_ball = batch["target_ball"].to(self.device)

        tokens = self.forward_backbone(t1, t2, b)   # [1,T,D]

        last = tokens[:, -1]   # [1,D]

        pred_ball = self.model.ball_head(last)

        loss = self.mse(pred_ball.squeeze(0), target_ball)

        return loss

    # ------------------------------------------------------

    # 2) masked player modeling
    def _step_masked(self, batch):

        t1 = batch["masked_team1"].unsqueeze(0).to(self.device)   # [1,T,11,4]
        t2 = batch["masked_team2"].unsqueeze(0).to(self.device)   # [1,T,11,4]

        B, T, _, _ = t1.shape

        # no ball is used for this task → feed zeros
        b = torch.zeros(B, T, 4, device=self.device)

        target_t1 = batch["target_team1"].to(self.device)
        target_t2 = batch["target_team2"].to(self.device)

        mask = batch["mask"].to(self.device)   # should be length 22
        
        tokens = self.forward_backbone(t1, t2, b)
        last = tokens[:, -1]   # [1,D]
        
        pred = self.model.masked_head(last).view(22, 2)   # positions only

        # target: positions only (cx, cy) from last frame
        target = torch.cat(
            [target_t1[-1, :, :2], target_t2[-1, :, :2]],
            dim=0
        )   # [22, 2]

        loss = self.mse(pred[mask], target[mask]) * 0.01

        return loss

    # ------------------------------------------------------

    # 3) possession continuity
    def _step_possession(self, batch):

        t1 = batch["team1"].unsqueeze(0).unsqueeze(0).to(self.device)
        t2 = batch["team2"].unsqueeze(0).unsqueeze(0).to(self.device)
        b  = batch["ball"].unsqueeze(0).unsqueeze(0).to(self.device)

        y = batch["label"].unsqueeze(0).to(self.device)

        tokens = self.forward_backbone(t1, t2, b)

        token = tokens[:, 0]

        logits = self.model.possession_head(token)

        loss = self.ce(logits, y)

        return loss

    # ------------------------------------------------------

    # 4) temporal order
    def _step_order(self, batch):

        t1_1 = batch["clip1_team1"].unsqueeze(0).to(self.device)
        t2_1 = batch["clip1_team2"].unsqueeze(0).to(self.device)
        b1   = batch["clip1_ball"].unsqueeze(0).to(self.device)

        t1_2 = batch["clip2_team1"].unsqueeze(0).to(self.device)
        t2_2 = batch["clip2_team2"].unsqueeze(0).to(self.device)
        b2   = batch["clip2_ball"].unsqueeze(0).to(self.device)

        y = torch.tensor([batch["label"]], device=self.device)

        z1 = self.forward_backbone(t1_1, t2_1, b1).mean(dim=1)
        z2 = self.forward_backbone(t1_2, t2_2, b2).mean(dim=1)

        pair = torch.cat([z1, z2], dim=-1)

        logits = self.model.order_head(pair)

        loss = self.ce(logits, y)

        return loss

    # ------------------------------------------------------

    # 5) contrastive
    def _step_contrastive(self, batch):

        t1_v1 = batch["view1_team1"].unsqueeze(0).to(self.device)
        t2_v1 = batch["view1_team2"].unsqueeze(0).to(self.device)
        b_v1  = batch["view1_ball"].unsqueeze(0).to(self.device)

        t1_v2 = batch["view2_team1"].unsqueeze(0).to(self.device)
        t2_v2 = batch["view2_team2"].unsqueeze(0).to(self.device)
        b_v2  = batch["view2_ball"].unsqueeze(0).to(self.device)

        z1 = self.forward_backbone(t1_v1, t2_v1, b_v1).mean(dim=1)
        z2 = self.forward_backbone(t1_v2, t2_v2, b_v2).mean(dim=1)

        z1 = nn.functional.normalize(z1, dim=-1)
        z2 = nn.functional.normalize(z2, dim=-1)

        # InfoNCE / NT-Xent with temperature — prevents collapse
        # change temperature from 0.07 to 0.5 — less aggressive pushing
        temperature = 0.5

        sim = torch.mm(z1, z2.T) / temperature   # (1,1) — works with batch_size=1 too

        # positive pair loss: z1[i] should be close to z2[i]
        # with batch_size=1 this simplifies to:
        loss = -nn.functional.logsigmoid(sim.squeeze())

        return loss