from env.simulator import SoccerEnv
from rl.agent import HierarchicalPPOAgent
from rl.trainer import update_agent, shaping_reward
from rl.buffer import ReplayBuffer
import torch
import numpy as np
import pygame
import time
import math
import matplotlib.pyplot as plt
from collections import deque

MACRO_ACTIONS = ["move", "shoot", "intercept"]
SHOOT_COOL_DOWN = 18
INTERCEPT_COOL_DOWN = 9

# Initialize Pygame for visualization
pygame.init()
WIDTH, HEIGHT = 800, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Soccer Simulator - Training Visualization")
clock = pygame.time.Clock()
SCALE = WIDTH / 100, HEIGHT / 60
FONT = pygame.font.SysFont(None, 24)

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 180, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BALL_COLOR = (220, 100, 0)
GRAY = (100, 100, 100)

def to_px(pos):
    return int(pos[0] * SCALE[0]), int(pos[1] * SCALE[1])

def draw_arrow(start, direction, agent_id="A", length=20, color=(255, 255, 0)):
    if not hasattr(draw_arrow, "last_dirs"):
        draw_arrow.last_dirs = {"A": np.array([1.0, 0.0]), "B": np.array([-1.0, 0.0])}
    
    norm = np.linalg.norm(direction)
    if norm >= 1e-6:
        unit = direction / norm
        draw_arrow.last_dirs[agent_id] = unit
    else:
        unit = draw_arrow.last_dirs[agent_id]
        
    end = (start[0] + length * unit[0], start[1] + length * unit[1])
    pygame.draw.line(screen, color, start, end, 3)

def render_env(env, macro_A, macro_B, elapsed_time, cum_A, cum_B, score_A, score_B, reward_window=None):
    goal_height = int(14 * SCALE[1])
    screen.fill(GREEN)
    pygame.draw.rect(screen, WHITE, pygame.Rect(0, 0, WIDTH, HEIGHT), 5)
    pygame.draw.line(screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2)
    pygame.draw.circle(screen, WHITE, (WIDTH // 2, HEIGHT // 2), int(10 * SCALE[0]), 2)

    # Draw goal boxes
    box_width = int(12 * SCALE[0])
    box_height = int(24 * SCALE[1])
    pygame.draw.rect(
        screen, WHITE, pygame.Rect(0, (HEIGHT - box_height) // 2, box_width, box_height), 2
    )
    pygame.draw.rect(
        screen, WHITE, pygame.Rect(WIDTH - box_width, (HEIGHT - box_height) // 2, box_width, box_height), 2
    )

    # Draw goals
    pygame.draw.polygon(
        screen,
        YELLOW,
        [
            (2, HEIGHT // 2 - goal_height // 2),
            (10, HEIGHT // 2),
            (2, HEIGHT // 2 + goal_height // 2),
        ],
    )
    pygame.draw.polygon(
        screen,
        YELLOW,
        [
            (WIDTH - 2, HEIGHT // 2 - goal_height // 2),
            (WIDTH - 10, HEIGHT // 2),
            (WIDTH - 2, HEIGHT // 2 + goal_height // 2),
        ],
    )

    # Draw agents
    pos_A = to_px(env.state[0:2])
    vel_A = env.state[2:4]
    pygame.draw.circle(screen, BLUE, pos_A, 20)
    draw_arrow(pos_A, vel_A, agent_id="A", color=YELLOW)

    pos_B = to_px(env.state[4:6])
    vel_B = env.state[6:8]
    pygame.draw.circle(screen, RED, pos_B, 20)
    draw_arrow(pos_B, vel_B, agent_id="B", color=YELLOW)

    # Draw ball with bouncing effect
    ball_px = to_px(env.state[8:10])
    bounce = 2 * math.sin(elapsed_time * 10)
    pygame.draw.circle(screen, BALL_COLOR, (ball_px[0], ball_px[1] - int(bounce + 6)), 6)

    # Display game info
    macro_str_A = MACRO_ACTIONS[macro_A] if macro_A >= 0 and macro_A < len(MACRO_ACTIONS) else "None"
    macro_str_B = MACRO_ACTIONS[macro_B] if macro_B >= 0 and macro_B < len(MACRO_ACTIONS) else "None"
    
    screen.blit(FONT.render(f"Agent A: {macro_str_A}", True, WHITE), (10, 10))
    screen.blit(FONT.render(f"Agent B: {macro_str_B}", True, WHITE), (10, 30))
    screen.blit(FONT.render(f"Ball dist A: {np.linalg.norm(env.state[0:2] - env.state[8:10]):.2f}", True, WHITE), (10, 50))
    screen.blit(FONT.render(f"Ball dist B: {np.linalg.norm(env.state[4:6] - env.state[8:10]):.2f}", True, WHITE), (10, 70))
    screen.blit(FONT.render(f"Ball owner: {env.ball_owner if env.ball_owner else 'None'}", True, WHITE), (10, 90))
    screen.blit(FONT.render(f"Ball vel: [{env.state[10]:.2f}, {env.state[11]:.2f}]", True, WHITE), (10, 110))
    
    screen.blit(FONT.render(f"Time: {elapsed_time:.1f}s", True, WHITE), (WIDTH - 120, 10))
    screen.blit(FONT.render(f"Reward A: {cum_A:.2f}", True, WHITE), (WIDTH - 180, 30))
    screen.blit(FONT.render(f"Reward B: {cum_B:.2f}", True, WHITE), (WIDTH - 180, 50))
    
    # Display score
    scoreboard = FONT.render(f"Score  A: {score_A}  |  B: {score_B}", True, WHITE)
    scoreboard_rect = scoreboard.get_rect(center=(WIDTH // 2, 20))
    screen.blit(scoreboard, scoreboard_rect)
    
    # If we have reward data, draw a small chart
    if reward_window is not None and len(reward_window) > 1:
        # Draw reward trend
        chart_width, chart_height = 200, 80
        chart_x, chart_y = WIDTH - chart_width - 10, HEIGHT - chart_height - 10
        
        # Background and border
        pygame.draw.rect(screen, (0, 0, 0, 128), (chart_x, chart_y, chart_width, chart_height))
        pygame.draw.rect(screen, WHITE, (chart_x, chart_y, chart_width, chart_height), 1)
        
        # Plot data
        max_val = max(max(reward_window), 1)
        min_val = min(min(reward_window), -1)
        range_val = max(max_val - min_val, 1)
        
        # Draw zero line if in range
        if min_val <= 0 and max_val >= 0:
            zero_y = chart_y + chart_height - int((0 - min_val) / range_val * chart_height)
            pygame.draw.line(screen, GRAY, (chart_x, zero_y), (chart_x + chart_width, zero_y), 1)
        
        # Draw reward data
        for i in range(1, len(reward_window)):
            x1 = chart_x + int((i-1) * chart_width / (len(reward_window) - 1))
            y1 = chart_y + chart_height - int((reward_window[i-1] - min_val) / range_val * chart_height)
            x2 = chart_x + int(i * chart_width / (len(reward_window) - 1))
            y2 = chart_y + chart_height - int((reward_window[i] - min_val) / range_val * chart_height)
            pygame.draw.line(screen, BLUE, (x1, y1), (x2, y2), 2)
        
        # Label
        screen.blit(FONT.render("Recent A Rewards", True, WHITE), (chart_x, chart_y - 25))

    pygame.display.flip()

def train_with_visualization(env, agent_A, agent_B, num_episodes, fps=30):
    buffer_A = ReplayBuffer()
    buffer_B = ReplayBuffer()
    reward_tracker = []
    MAX_STEPS = 1000
    reward_window = deque(maxlen=100)  # For visualizing recent rewards
    
    score_A = score_B = 0
    
    print("[INFO] Starting visual training...")
    
    running = True
    for ep in range(num_episodes):
        if not running:
            break
            
        state = env.reset(training=True)
        done = False
        episode_reward_A = 0
        episode_reward_B = 0
        step_count = 0
        macro_history_A = []
        macro_history_B = []
        start_time = time.time()
        
        macro_A = macro_B = 0
        param_A = param_B = [0.0, 0.0]
        
        while not done and step_count < MAX_STEPS and running:
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    elif event.key == pygame.K_SPACE:
                        # Space to pause/resume
                        paused = True
                        while paused and running:
                            for pause_event in pygame.event.get():
                                if pause_event.type == pygame.QUIT:
                                    running = False
                                    paused = False
                                elif pause_event.type == pygame.KEYDOWN:
                                    if pause_event.key == pygame.K_ESCAPE:
                                        running = False
                                        paused = False
                                    elif pause_event.key == pygame.K_SPACE:
                                        paused = False
                            
                            screen.blit(FONT.render("PAUSED - Press SPACE to continue", True, WHITE), (WIDTH // 2 - 150, HEIGHT // 2))
                            pygame.display.flip()
                            clock.tick(10)
            
            if not running:
                break
            
            # Agent A's action
            if env.cooldowns["A"] > 0:
                macro_A, param_A = -1, [0.0, 0.0]
            else:
                macro_A, param_A = agent_A.select_action(state)
            
            # Agent B's action
            if env.cooldowns["B"] > 0:
                macro_B, param_B = -1, [0.0, 0.0]
            else:
                macro_B, param_B = agent_B.select_action(state)
            
            # Process Agent A's action
            if macro_A != -1:
                if MACRO_ACTIONS[macro_A] == "shoot" and env.cooldowns["A"] == SHOOT_COOL_DOWN:
                    reward = 50 if env.ball_owner == "A" else -100
                    buffer_A.add((state.copy(), macro_A, param_A, reward, state.copy(), done))
                    episode_reward_A += reward
                elif MACRO_ACTIONS[macro_A] == "intercept" and env.cooldowns["A"] == INTERCEPT_COOL_DOWN:
                    agent_pos = state[0:2]
                    ball_pos = state[8:10]
                    dist = np.linalg.norm(agent_pos - ball_pos)
                    reward = -80 if env.ball_owner == "A" else (-20 if dist >= 10 else 0)
                    if reward != 0:
                        buffer_A.add((state.copy(), macro_A, param_A, reward, state.copy(), done))
                        episode_reward_A += reward
                macro_history_A.append((state.copy(), macro_A, param_A))
            
            # Process Agent B's action
            if macro_B != -1:
                if MACRO_ACTIONS[macro_B] == "shoot" and env.cooldowns["B"] == SHOOT_COOL_DOWN:
                    reward = 50 if env.ball_owner == "B" else -100
                    buffer_B.add((state.copy(), macro_B, param_B, reward, state.copy(), done))
                    episode_reward_B += reward
                elif MACRO_ACTIONS[macro_B] == "intercept" and env.cooldowns["B"] == INTERCEPT_COOL_DOWN:
                    agent_pos = state[4:6]
                    ball_pos = state[8:10]
                    dist = np.linalg.norm(agent_pos - ball_pos)
                    reward = -80 if env.ball_owner == "B" else (-20 if dist >= 10 else 0)
                    if reward != 0:
                        buffer_B.add((state.copy(), macro_B, param_B, reward, state.copy(), done))
                        episode_reward_B += reward
                macro_history_B.append((state.copy(), macro_B, param_B))
            
            # Take a step in the environment
            next_state, r_A, r_B, done = env.step(macro_A, param_A, macro_B, param_B)
            
            # Update scores if goals were scored
            if r_A > 20:  # Goal for A
                score_A += 1
            if r_B > 20:  # Goal for B
                score_B += 1
            
            # Process delayed rewards
            if r_A != 0 and macro_history_A:
                s, m, p = macro_history_A.pop(0)
                buffer_A.add((s, m, p, r_A, next_state.copy(), done))
                episode_reward_A += r_A
                reward_window.append(r_A)  # Add to visualization window
                
            if r_B != 0 and macro_history_B:
                s, m, p = macro_history_B.pop(0)
                buffer_B.add((s, m, p, r_B, next_state.copy(), done))
                episode_reward_B += r_B
            
            # Process move rewards through shaping
            if macro_A == 0:
                move_r = shaping_reward(state, agent="A")
                buffer_A.add((state.copy(), macro_A, param_A, move_r, next_state.copy(), done))
                episode_reward_A += move_r
                
            if macro_B == 0:
                move_r = shaping_reward(state, agent="B")
                buffer_B.add((state.copy(), macro_B, param_B, move_r, next_state.copy(), done))
                episode_reward_B += move_r
            
            # Visualize current state
            elapsed_time = time.time() - start_time
            render_env(env, macro_A, macro_B, elapsed_time, 
                      episode_reward_A, episode_reward_B, 
                      score_A, score_B, 
                      reward_window)
            
            # Control frame rate
            clock.tick(fps)
            
            state = next_state
            step_count += 1
        
        print(f"[EP {ep}] Finished in {step_count} steps | Total Reward A: {episode_reward_A:.2f} | B: {episode_reward_B:.2f}")
        
        # Update agents based on collected experiences
        update_agent(agent_A, buffer_A.sample())
        update_agent(agent_B, buffer_B.sample())
        buffer_A.clear()
        buffer_B.clear()
        
        reward_tracker.append((episode_reward_A, episode_reward_B))
        
        # Save checkpoints
        if (ep + 1) % 25 == 0:
            torch.save(agent_A.policy.state_dict(), f"checkpoint_A_ep{ep+1}.pth")
            torch.save(agent_B.policy.state_dict(), f"checkpoint_B_ep{ep+1}.pth")
    
    print("[INFO] Training completed.")
    pygame.quit()
    
    # Plot final learning curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot([x[0] for x in reward_tracker], label="Agent A")
    plt.plot([x[1] for x in reward_tracker], label="Agent B")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    window_size = min(10, len(reward_tracker))
    if window_size > 0:
        smoothed_A = [sum(x[0] for x in reward_tracker[i:i+window_size])/window_size 
                     for i in range(len(reward_tracker)-window_size+1)]
        smoothed_B = [sum(x[1] for x in reward_tracker[i:i+window_size])/window_size 
                     for i in range(len(reward_tracker)-window_size+1)]
        plt.plot(smoothed_A, label="Agent A (Smoothed)")
        plt.plot(smoothed_B, label="Agent B (Smoothed)")
        plt.xlabel("Episode")
        plt.ylabel("Avg Reward (Window Size 10)")
        plt.title("Smoothed Training Progress")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.show()
    
    return reward_tracker

if __name__ == "__main__":
    env = SoccerEnv()
    agent_A = HierarchicalPPOAgent(name="A")
    agent_B = HierarchicalPPOAgent(name="B")
    
    # Start training with visualization
    fps = 60  # Higher value for faster training, lower for better visualization
    reward_log = train_with_visualization(env, agent_A, agent_B, num_episodes=100, fps=fps)
    
    # Save trained models
    torch.save(agent_A.policy.state_dict(), "agent_A.pth")
    torch.save(agent_B.policy.state_dict(), "agent_B.pth")
    
    print("Training complete. Models saved.")