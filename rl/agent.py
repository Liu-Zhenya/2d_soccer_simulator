import torch
import torch.nn as nn
import numpy as np
import math

MACRO_ACTIONS = ["move", "shoot", "intercept"]
MACRO_ACTIONS_WO = ["move"]


class MacroPolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, agent = "A"):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.macro_head_withball = nn.Linear(hidden_dim, len(MACRO_ACTIONS))
        self.macro_head_without = nn.Linear(hidden_dim, len(MACRO_ACTIONS_WO))
        self.agent = agent
        self.param_heads_withball = nn.ModuleDict(
            {
                "move": nn.Linear(hidden_dim, 2),  # [angle, magnitude]
                "shoot": nn.Linear(hidden_dim, 1),  # [angle]
                "intercept": nn.Linear(hidden_dim, 2),  # [angle, magnitude]
            }
        )

        self.param_heads_withoutball = nn.ModuleDict(
                {
                    "move": nn.Linear(hidden_dim, 2),  # [angle, magnitude]
                }
            )

    def forward(self, state):
        """
        Returns:
            - shared features
            - macro logits (before softmax)
            - parameter outputs for each macro
        """
        x = self.shared(state)

        # Determine ownership mask for each sample
        if self.agent == "A":
            has_ball = state[:, 12] == 1
        else:
            has_ball = state[:, 13] == 1

        # Prepare output tensors
        macro_logits = torch.zeros((state.shape[0], len(MACRO_ACTIONS)), device=state.device)
        param_outputs = {}

        # Process samples with ball
        if has_ball.any():
            x_with = x[has_ball]
            logits_with = self.macro_head_withball(x_with)
            macro_logits[has_ball] = logits_with

            for name, head in self.param_heads_withball.items():
                out = head(x_with)
                if name not in param_outputs:
                    param_outputs[name] = torch.zeros((state.shape[0], out.shape[1]), device=state.device)
                param_outputs[name][has_ball] = out

        # Process samples without ball
        if (~has_ball).any():
            x_without = x[~has_ball]
            logits_without = self.macro_head_without(x_without)
            macro_logits[~has_ball, :logits_without.shape[1]] = self.macro_head_without(x_without)

            for name, head in self.param_heads_withoutball.items():
                out = head(x_without)
                if name not in param_outputs:
                    param_outputs[name] = torch.zeros((state.shape[0], out.shape[1]), device=state.device)
                param_outputs[name][~has_ball] = out

        return x, macro_logits, param_outputs
#             mag = torch.sigmoid(raw[1]) * 1.5



class HierarchicalPPOAgent:
    def __init__(self, name, input_dim=14, lr=1e-3):
        self.name = name
        self.policy = MacroPolicyNet(input_dim, agent=name)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state_np, return_probs=False):
        self.policy.eval()
        state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            _, macro_logits, param_outputs = self.policy(state)

        macro_probs = torch.softmax(macro_logits, dim=-1).squeeze(0).numpy()
        if (self.name == "A" and state_np[12] == 0) or (self.name == "B" and state_np[13] == 0):
            macro = np.random.choice(len(MACRO_ACTIONS_WO), p=[1.0])
            macro_str = MACRO_ACTIONS_WO[macro]
            raw_param = param_outputs[macro_str].squeeze(0)
        elif (self.name == "A" and state_np[12] == 1) or (self.name == "B" and state_np[13] == 1):
            macro = np.random.choice(len(MACRO_ACTIONS), p=macro_probs)
            macro_str = MACRO_ACTIONS[macro]
            raw_param = param_outputs[macro_str].squeeze(0)

        if macro_str == "shoot":
            angle = torch.tanh(raw_param[0]) * math.pi
            param = [angle.item(), 1.0]
        else:
            angle = torch.tanh(raw_param[0]) * math.pi
            mag = torch.sigmoid(raw_param[1]) * 1.5
            param = [angle.item(), mag.item()]

        if return_probs:
            return macro, param, macro_probs
        else:
            return macro, param
