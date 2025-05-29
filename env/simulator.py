import numpy as np

FRICTION = 0.98
FIELD_WIDTH = 100
FIELD_HEIGHT = 60
STEAL_RANGE = 10.0
BALL_SPEED_DECAY = 0.98
GOAL_WIDTH = 14
AGENT_RADIUS = 3.0
SHOOT_COOL_DOWN = 18
INTERCEPT_COOL_DOWN = 2
MACRO_ACTIONS = ["move", "shoot", "intercept"]


class SoccerEnv:
    def __init__(self):
        self.prev_macro_action_A = 0
        self.prev_macro_action_B = 0
        self.last_reset_ball_owner = None
        self.episode_count = 0
        self.training_mode = False
        self.event_table = []
        self.pending_rewards = []
        self.cooldowns = {"A": 0, "B": 0}
        self.prev_ball_pos = np.zeros(2)
        self.ball_owner = None
        self.who_start_game = None
        self.reset()

    def reset(self, training=True):
        self.training_mode = training
        self.state = np.zeros(12)
        self.state[10:12] = [0, 0]
        self.done = False
        self.t = 0
        self.event_table.clear()
        self.pending_rewards.clear()
        self.cooldowns = {"A": 0, "B": 0}

        if self.training_mode:
            # self.who_start_game = "A" if self.episode_count % 2 == 0 else "B"
            self.who_to_start_game = "B"
        else:
            # no meaning right now
            self.who_start_game = self.last_reset_ball_owner

        if self.who_start_game == "A":
            self.state[0:2] = [45, 30]
            self.state[2:4] = [1.5, 0]
            self.state[4:6] = [80, 30]
            self.state[6:8] = [-1.5, 0]
            self.state[8:10] = self.state[0:2] + np.array([AGENT_RADIUS, 0])
        elif self.who_start_game == "B":
            self.state[4:6] = [55, 30]
            self.state[6:8] = [-1.5, 0]
            self.state[0:2] = [20, 30]
            self.state[2:4] = [1.5, 0]
            self.state[8:10] = self.state[4:6] + np.array([-AGENT_RADIUS, 0])
        else:
            self.state[0:2] = [40, 30]
            self.state[2:4] = [1.5, 0]
            self.state[4:6] = [80, 30]
            self.state[6:8] = [-1.5, 0]
            self.state[8:10] = [45, 30]

        self.prev_ball_pos = self.state[8:10].copy()
        self.episode_count += 1
        # self.last_reset_ball_owner = self.ball_owner
        return self.state.copy()

    def step(self, action_A, param_A, action_B, param_B):
        self.prev_ball_pos = self.state[8:10].copy()

        if self.cooldowns["A"] > 0:
            action_A, param_A = -1, [0, 0]
            self.cooldowns["A"] -= 1
        if self.cooldowns["B"] > 0:
            action_B, param_B = -1, [0, 0]
            self.cooldowns["B"] -= 1

        self.prev_macro_action_A = action_A
        self.prev_macro_action_B = action_B

        self._resolve_events(action_A, param_A, action_B, param_B)
        self._update_agents(action_A, param_A, action_B, param_B)
        self._update_ball()
        # self._check_game_end_conditions()

        reward_A = self._compute_reward("A")
        reward_B = self._compute_reward("B")

        self.t += 1
        return self.state.copy(), reward_A, reward_B, self.done

    def _macro_name(self, idx):
        return MACRO_ACTIONS[idx] if 0 <= idx < len(MACRO_ACTIONS) else None

    def _update_agents(self, macro_A, param_A, macro_B, param_B):
        for macro, param, offset, label in zip(
            [macro_B], [param_B], [4], ["B"]
        ):
            if macro == -1:
                continue
            if self._macro_name(macro) == "shoot":
                self.event_table.append(
                    {
                        "type": "shoot",
                        "agent": label,
                        "param": param,
                        "start_time": self.t,
                        "status": "pending",
                    }
                )
                self.cooldowns[label] = SHOOT_COOL_DOWN
            elif self._macro_name(macro) == "intercept":
                self.event_table.append(
                    {
                        "type": "intercept",
                        "agent": label,
                        "param": param,
                        "start_time": self.t,
                        "status": "pending",
                    }
                )
                self.cooldowns[label] = INTERCEPT_COOL_DOWN
            elif self._macro_name(macro) == "move":
                direction = np.array([np.cos(param[0]), np.sin(param[0])]) * param[1] * 1.5
                self.state[offset + 2 : offset + 4] = direction
                new_pos = self.state[offset : offset + 2] + direction
                new_pos[0] = np.clip(new_pos[0], 0, FIELD_WIDTH)
                new_pos[1] = np.clip(new_pos[1], 0, FIELD_HEIGHT)
                self.state[offset : offset + 2] = new_pos
                if self.ball_owner == label:
                    face = direction / (np.linalg.norm(direction) + 1e-6)
                    self.state[8:10] = new_pos + face * AGENT_RADIUS

    def _update_ball(self):
        for e in self.event_table:
            if e["status"] == "pending" and self.t - e["start_time"] == SHOOT_COOL_DOWN:
                if e["type"] == "shoot" and self.ball_owner == e["agent"]:
                    angle = e["param"][0]
                    shoot_dir = np.array([np.cos(angle), np.sin(angle)])
                    power = 1.0  # fixed power
                    self.state[10:12] = shoot_dir * power * 5.0
                    self.ball_owner = None

        self.state[10:12] *= BALL_SPEED_DECAY
        self.state[8:10] += self.state[10:12]

        # for offset, label in [(0, "A"), (4, "B")]:
        for offset, label in [(4, "B")]:
            if self.ball_owner is None:
                agent_pos = self.state[offset : offset + 2]
                dist = np.linalg.norm(agent_pos - self.state[8:10])
                if dist < 2.0:
                    self.ball_owner = label

    def _resolve_events(self, macro_A, param_A, macro_B, param_B):
        """
        Places define the rewards
        """
        for e in self.event_table:
            if e["status"] == "pending":
                elapsed = self.t - e["start_time"]

                if e["type"] == "shoot" and elapsed >= 18:
                    goal_y = self.state[9]
                    curr_x = self.state[8]

                    if curr_x <= 0 and 23 <= goal_y <= 37:
                        self.pending_rewards.append(("B", 500))
                        self.pending_rewards.append(("A", -50))
                        self.done = True
                        e["status"] = "done"
                        self.event_table.append(
                            {"type": "goal", "agent": "B", "status": "done"}
                        )
                        print(f"[Resolved] Shoot by {e['agent']} → HIT THE GOAL")

                    elif curr_x >= FIELD_WIDTH and 23 <= goal_y <= 37:
                        self.pending_rewards.append(("A", 100))
                        self.pending_rewards.append(("B", -50))
                        self.done = True
                        e["status"] = "done"
                        self.event_table.append(
                            {"type": "goal", "agent": "A", "status": "done"}
                        )

                    elif (
                        self.state[8] < 0
                        or self.state[8] > FIELD_WIDTH
                        or self.state[9] < 0
                        or self.state[9] > FIELD_HEIGHT
                    ):
                        self.pending_rewards.append((e["agent"], 0))
                        print(f"[Resolved] Shoot by {e['agent']} → MISS at t={self.t}")
                        e["status"] = "done"
                        self.done = True

                elif e["type"] == "intercept" and elapsed >= INTERCEPT_COOL_DOWN:
                    agent_offset = 0 if e["agent"] == "A" else 4
                    agent_pos = self.state[agent_offset : agent_offset + 2]
                    ball_pos = self.state[8:10]
                    dist = np.linalg.norm(agent_pos - ball_pos)

                    if self.ball_owner == e["agent"]:
                        self.pending_rewards.append((e["agent"], -80))
                    elif dist < STEAL_RANGE:
                        self.ball_owner = e["agent"]
                        self.pending_rewards.append((e["agent"], 100))
                    else:
                        self.pending_rewards.append((e["agent"], -20))

                    e["status"] = "done"

            # if e["status"] == "done":
            #     self.event_table.remove(e)

    # def _check_game_end_conditions(self):
    #     x, y = self.state[8:10]
    #     goal_y_min = FIELD_HEIGHT / 2 - 7
    #     goal_y_max = FIELD_HEIGHT / 2 + 7

    #     if x >= FIELD_WIDTH and goal_y_min <= y <= goal_y_max:
    #         self.done = True
    #     elif x <= 0 and goal_y_min <= y <= goal_y_max:
    #         self.done = True
    #     elif y < 0 or y > FIELD_HEIGHT or x < 0 or x > FIELD_WIDTH:
    #         self.done = True

    def _compute_reward(self, agent):
        reward = 0.0
        new_pending = []
        for a, r in self.pending_rewards:
            if a == agent:
                reward += r
            else:
                new_pending.append((a, r))
        self.pending_rewards = new_pending
        return reward
