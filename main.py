from env.simulator import SoccerEnv
from rl.agent import HierarchicalPPOAgent
from rl.trainer import train_self_play
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    env = SoccerEnv()
    agent_A = HierarchicalPPOAgent(name="A")
    agent_B = HierarchicalPPOAgent(name="B")

    reward_log = train_self_play(env, agent_A, agent_B, num_episodes=100)

    # Save trained models
    torch.save(agent_A.policy.state_dict(), "agent_A.pth")
    torch.save(agent_B.policy.state_dict(), "agent_B.pth")

    # Plot learning curve
    plt.plot(reward_log)
    plt.xlabel("Episode (per 100)")
    plt.ylabel("Average Reward (A)")
    plt.title("Learning Curve of Agent A")
    plt.grid(True)
    plt.show()
