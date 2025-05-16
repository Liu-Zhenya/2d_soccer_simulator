import torch
import torch.nn as nn
import numpy as np
import math

MACRO_ACTIONS = ["move", "shoot", "intercept"]


class MacroPolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.macro_head = nn.Linear(hidden_dim, len(MACRO_ACTIONS))

        self.param_heads = nn.ModuleDict(
            {
                "move": nn.Linear(hidden_dim, 2),  # [angle, magnitude]
                "shoot": nn.Linear(hidden_dim, 1),  # [angle]
                "intercept": nn.Linear(hidden_dim, 2),  # [angle, magnitude]
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
        macro_logits = self.macro_head(x)
        param_outputs = {name: head(x) for name, head in self.param_heads.items()}
        return x, macro_logits, param_outputs


class HierarchicalPPOAgent:
    def __init__(self, name, input_dim=12, lr=1e-3):
        self.name = name
        self.policy = MacroPolicyNet(input_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state_np, return_probs=False):
        self.policy.eval()
        state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            _, macro_logits, param_outputs = self.policy(state)

        macro_probs = torch.softmax(macro_logits, dim=-1).squeeze(0).numpy()
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
