from rl.buffer import ReplayBuffer
import torch
import torch.nn.functional as F
import numpy as np
import math

MACRO_ACTIONS = ["move", "shoot", "intercept"]
SHOOT_COOL_DOWN = 18
INTERCEPT_COOL_DOWN = 9


def train_self_play(env, agent_A, agent_B, num_episodes):
    buffer_A = ReplayBuffer()
    buffer_B = ReplayBuffer()
    reward_tracker = []
    MAX_STEPS = 100

    print("[INFO] Starting self-play training...")

    for ep in range(num_episodes):
        state = env.reset(training=True)
        done = False
        episode_reward_A = 0
        episode_reward_B = 0
        step_count = 0
        macro_history_A = []
        macro_history_B = []

        while not done and step_count < MAX_STEPS:
            # Always set Agent A's action to -1 (no action) to ban movement
            macro_A, param_A = -1, [0.0, 0.0]

            if env.cooldowns["B"] > 0:
                macro_B, param_B = -1, [0.0, 0.0]
            else:
                macro_B, param_B = agent_B.select_action(state)

            # Skip processing Agent A's actions since they're always -1 now
            # We keep this commented out for reference
            # if macro_A != -1:
            #     if (
            #         MACRO_ACTIONS[macro_A] == "shoot"
            #         and env.cooldowns["A"] == SHOOT_COOL_DOWN
            #     ):
            #         reward = 50 if env.ball_owner == "A" else -100
            #         buffer_A.add(
            #             (state.copy(), macro_A, param_A, reward, state.copy(), done)
            #         )
            #         episode_reward_A += reward
            #     elif (
            #         MACRO_ACTIONS[macro_A] == "intercept"
            #         and env.cooldowns["A"] == INTERCEPT_COOL_DOWN
            #     ):
            #         agent_pos = state[0:2]
            #         ball_pos = state[8:10]
            #         dist = np.linalg.norm(agent_pos - ball_pos)
            #         reward = -80 if env.ball_owner == "A" else (-20 if dist >= 10 else 0)
            #         if reward != 0:
            #             buffer_A.add(
            #                 (state.copy(), macro_A, param_A, reward, state.copy(), done)
            #             )
            #             episode_reward_A += reward
            #     macro_history_A.append((state.copy(), macro_A, param_A))

            if macro_B != -1:
                if (
                    MACRO_ACTIONS[macro_B] == "shoot"
                    and env.cooldowns["B"] == SHOOT_COOL_DOWN
                ):
                    reward = 50 if env.ball_owner == "B" else -100
                    buffer_B.add(
                        (state.copy(), macro_B, param_B, reward, state.copy(), done)
                    )
                    episode_reward_B += reward
                elif (
                    MACRO_ACTIONS[macro_B] == "intercept"
                    and env.cooldowns["B"] == INTERCEPT_COOL_DOWN
                ):
                    agent_pos = state[4:6]
                    ball_pos = state[8:10]
                    dist = np.linalg.norm(agent_pos - ball_pos)
                    reward = -80 if env.ball_owner == "B" else (-20 if dist >= 10 else 0)
                    if reward != 0:
                        buffer_B.add(
                            (state.copy(), macro_B, param_B, reward, state.copy(), done)
                        )
                        episode_reward_B += reward
                macro_history_B.append((state.copy(), macro_B, param_B))

            next_state, r_A, r_B, done = env.step(macro_A, param_A, macro_B, param_B)

            # Skip processing delayed rewards for Agent A since it doesn't take actions
            # if r_A != 0 and macro_history_A:
            #     s, m, p = macro_history_A.pop(0)
            #     buffer_A.add((s, m, p, r_A, next_state.copy(), done))
            #     episode_reward_A += r_A
            if r_B != 0 and macro_history_B:
                s, m, p = macro_history_B.pop(0)
                buffer_B.add((s, m, p, r_B, next_state.copy(), done))
                episode_reward_B += r_B

            # Skip move rewards for Agent A
            # if macro_A == 0:
            #     move_r = shaping_reward(state, agent="A")
            #     buffer_A.add(
            #         (state.copy(), macro_A, param_A, move_r, next_state.copy(), done)
            #     )
            #     episode_reward_A += move_r
            if macro_B == 0:
                move_r = shaping_reward(state, agent="B")
                buffer_B.add(
                    (state.copy(), macro_B, param_B, move_r, next_state.copy(), done)
                )
                episode_reward_B += move_r
            
            if (state[0] <= 2 or state[0]>=98 or state[1] >= 58 or state[1]<=2):
                env.done = True
                done = True

            state = next_state
            step_count += 1

        print(
            f"[EP {ep}] Finished in {step_count} steps | Total Reward A: {episode_reward_A:.2f} | B: {episode_reward_B:.2f}"
        )

        # Skip updating Agent A to ban its learning
        # update_agent(agent_A, buffer_A.sample())
        update_agent(agent_B, buffer_B.sample())
        buffer_A.clear()
        buffer_B.clear()

        reward_tracker.append((episode_reward_A, episode_reward_B))

    print("[INFO] Training completed.")
    return reward_tracker


def shaping_reward(state, agent):
    idx = 0 if agent == "A" else 4
    agent_pos = state[idx : idx + 2]
    ball_pos = state[8:10]

    ball_dist = np.linalg.norm(agent_pos - ball_pos)
    if ball_dist > 10.0:
        return -1.0

    # return 0.5 if ball_dist <= 3.0 else 0.9 * (1 / ball_dist)
    return 0.5 if ball_dist <= 3.0 else 0


def update_agent(agent, transitions):
    if not transitions:
        return

    states, macro_actions, action_params, rewards, next_states, dones = zip(*transitions)

    states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
    macro_tensor = torch.tensor(macro_actions, dtype=torch.long).unsqueeze(1)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)

    # Pad param_tensor to shape (N, 2)
    fixed_params = []
    for m, p in zip(macro_actions, action_params):
        macro_str = MACRO_ACTIONS[m]
        if macro_str == "shoot":
            fixed_params.append([p[0], 1.0])  # dummy mag
        else:
            fixed_params.append(p)
    param_tensor = torch.tensor(fixed_params, dtype=torch.float32)

    _, macro_logits, param_outputs = agent.policy(states_tensor)

    if torch.isnan(macro_logits).any():
        print("[ERROR] NaN in macro_logits")
        return

    macro_log_probs = torch.log_softmax(macro_logits, dim=-1).gather(1, macro_tensor)

    # Parameter log probs
    param_log_probs = []
    for i in range(len(states)):
        macro_str = MACRO_ACTIONS[macro_actions[i]]
        raw = param_outputs[macro_str][i]

        if macro_str == "shoot":
            angle = torch.tanh(raw[0]) * math.pi
            angle_dist = torch.distributions.Normal(angle, 1.0)
            angle_logp = angle_dist.log_prob(param_tensor[i][0])
            param_log_probs.append(angle_logp)
        else:
            angle = torch.tanh(raw[0]) * math.pi
            mag = torch.sigmoid(raw[1]) * 1.5
            angle_dist = torch.distributions.Normal(angle, 1.0)
            mag_dist = torch.distributions.Normal(mag, 0.2)
            angle_logp = angle_dist.log_prob(param_tensor[i][0])
            mag_logp = mag_dist.log_prob(param_tensor[i][1])
            param_log_probs.append(angle_logp + mag_logp)

    param_log_probs = torch.stack(param_log_probs).unsqueeze(1)
    total_log_probs = macro_log_probs + param_log_probs

    advantages = reward_tensor - reward_tensor.mean()
    if advantages.std() > 1e-6:
        advantages /= advantages.std()

    entropy_macro = (
        torch.distributions.Categorical(logits=macro_logits).entropy().unsqueeze(1)
    )
    loss = -(total_log_probs * advantages).mean() - 0.01 * entropy_macro.mean()

    agent.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), max_norm=1.0)
    agent.optimizer.step()

    print(f"[INFO] Updated agent {agent.name} | PPO Loss: {loss.item():.4f}")
