import torch
import torch.nn as nn
import numpy as np
import math

MACRO_ACTIONS = ["move", "shoot", "intercept"]
MACRO_ACTIONS_WO = ["move", "intercept"]
MACRO_ACTIONS_W = ["move", "shoot"]

class MacroPolicyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, agent = "A"):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.macro_head_withball = nn.Linear(hidden_dim, len(MACRO_ACTIONS_W))
        self.macro_head_without = nn.Linear(hidden_dim, len(MACRO_ACTIONS_WO))
        self.agent = agent
        self.param_heads_withball = nn.ModuleDict(
            {
                "move": nn.Linear(hidden_dim, 2),  # [angle, magnitude]
                "shoot": nn.Linear(hidden_dim, 1),  # [angle]
            }
        )

        self.param_heads_withoutball = nn.ModuleDict(
                {
                    "move": nn.Linear(hidden_dim, 2),  # [angle, magnitude]
                    "intercept": nn.Linear(hidden_dim, 2),  # [angle, magnitude]
                }
            )
    #     self._init_weights()

    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)

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
        zero = torch.tensor([-torch.inf], device=state.device, dtype=state.dtype)
        # Process samples with ball
        if has_ball.any():
            x_with = x[has_ball]
            logits_with = self.macro_head_withball(x_with)
            macro_logits[has_ball] = torch.stack((logits_with[:,0],logits_with[:,1], zero.expand(logits_with.size(0))),dim=1)

            for name, head in self.param_heads_withball.items():
                out = head(x_with)
                if name not in param_outputs:
                    param_outputs[name] = torch.zeros((state.shape[0], out.shape[1]), device=state.device)
                param_outputs[name][has_ball] = out

        # Process samples without ball
        if (~has_ball).any():
            x_without = x[~has_ball]
            logits_without = self.macro_head_without(x_without)
            macro_logits[~has_ball] = torch.stack((logits_without[:,0],zero.expand(logits_without.size(0)),logits_without[:,1]),dim=1)

            for name, head in self.param_heads_withoutball.items():
                out = head(x_without)
                if name not in param_outputs:
                    param_outputs[name] = torch.zeros((state.shape[0], out.shape[1]), device=state.device)
                param_outputs[name][~has_ball] = out
                macro_logits 

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
            macro = np.random.choice(len(MACRO_ACTIONS), p=macro_probs)
            macro_str = MACRO_ACTIONS[macro]
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
        
        if macro_str == "move":
            mag = torch.clamp(mag, min=0.5)

        if return_probs:
            return macro, param, macro_probs
        else:
            return macro, param
