import pygame
import numpy as np
import math
import time
import torch
from env.simulator import SoccerEnv
from rl.agent import HierarchicalPPOAgent
from rl.buffer import ReplayBuffer

MACRO_ACTIONS = ["move", "shoot", "intercept"]
SHOOT_COOL_DOWN = 18
INTERCEPT_COOL_DOWN = 9

pygame.init()
WIDTH, HEIGHT = 800, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
SCALE = WIDTH / 100, HEIGHT / 60
FONT = pygame.font.SysFont(None, 24)

WHITE = (255, 255, 255)
GREEN = (0, 180, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BALL_COLOR = (220, 100, 0)


def to_px(pos):
    return int(pos[0] * SCALE[0]), int(pos[1] * SCALE[1])


def draw_arrow(start, direction, agent_id="A", length=20, color=(255, 255, 0)):
    if not hasattr(draw_arrow, "last_dirs"):
        draw_arrow.last_dirs = {"A": np.array([1.0, 0.0]), "B": np.array([-1.0, 0.0])}
    norm = np.linalg.norm(direction)
    unit = direction / norm if norm >= 1e-6 else draw_arrow.last_dirs[agent_id]
    draw_arrow.last_dirs[agent_id] = unit
    end = (start[0] + length * unit[0], start[1] + length * unit[1])
    pygame.draw.line(screen, color, start, end, 3)


def shaping_reward(state, agent):
    idx = 0 if agent == "A" else 4
    agent_pos = state[idx : idx + 2]
    ball_pos = state[8:10]
    dist = np.linalg.norm(agent_pos - ball_pos)
    # return 0.5 if dist <= 3.0 else 0.9 * (1 / dist) if dist < 10.0 else -1.0
    return 0.5 if dist <= 3.0 else 0


def render(env, macro_str_A, macro_str_B, elapsed_time, cum_A, cum_B, score_A, score_B):
    screen.fill(GREEN)
    pygame.draw.rect(screen, WHITE, pygame.Rect(0, 0, WIDTH, HEIGHT), 5)
    pygame.draw.line(screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2)
    pygame.draw.circle(screen, WHITE, (WIDTH // 2, HEIGHT // 2), int(10 * SCALE[0]), 2)

    box_w, box_h = int(12 * SCALE[0]), int(24 * SCALE[1])
    pygame.draw.rect(screen, WHITE, pygame.Rect(0, (HEIGHT - box_h) // 2, box_w, box_h), 2)
    pygame.draw.rect(
        screen, WHITE, pygame.Rect(WIDTH - box_w, (HEIGHT - box_h) // 2, box_w, box_h), 2
    )

    pygame.draw.polygon(
        screen, YELLOW, [(2, HEIGHT // 2 - 42), (10, HEIGHT // 2), (2, HEIGHT // 2 + 42)]
    )
    pygame.draw.polygon(
        screen,
        YELLOW,
        [
            (WIDTH - 2, HEIGHT // 2 - 42),
            (WIDTH - 10, HEIGHT // 2),
            (WIDTH - 2, HEIGHT // 2 + 42),
        ],
    )

    for i, (color, offset, label) in enumerate([(BLUE, 0, "A"), (RED, 4, "B")]):
        pos = to_px(env.state[offset : offset + 2])
        vel = env.state[offset + 2 : offset + 4]
        pygame.draw.circle(screen, color, pos, 20)
        draw_arrow(pos, vel, agent_id=label, color=YELLOW)

    ball_px = to_px(env.state[8:10])
    bounce = 2 * math.sin(elapsed_time * 10)
    pygame.draw.circle(screen, BALL_COLOR, (ball_px[0], ball_px[1] - int(bounce + 6)), 6)

    screen.blit(FONT.render(f"Agent A: {macro_str_A}", True, WHITE), (10, 10))
    screen.blit(FONT.render(f"Agent B: {macro_str_B}", True, WHITE), (10, 30))
    screen.blit(
        FONT.render(
            f"Ball dist A: {np.linalg.norm(env.state[0:2] - env.state[8:10]):.2f}",
            True,
            WHITE,
        ),
        (10, 50),
    )
    screen.blit(
        FONT.render(
            f"Ball dist B: {np.linalg.norm(env.state[4:6] - env.state[8:10]):.2f}",
            True,
            WHITE,
        ),
        (10, 70),
    )
    screen.blit(
        FONT.render(
            f"Ball owner: {env.ball_owner if env.ball_owner else 'None'}", True, WHITE
        ),
        (10, 90),
    )
    screen.blit(
        FONT.render(f"Ball vel: [{env.state[10]:.2f}, {env.state[11]:.2f}]", True, WHITE),
        (10, 110),
    )
    screen.blit(FONT.render(f"Time: {elapsed_time:.1f}s", True, WHITE), (WIDTH - 120, 10))
    screen.blit(FONT.render(f"Reward A: {cum_A:.2f}", True, WHITE), (WIDTH - 180, 30))
    screen.blit(FONT.render(f"Reward B: {cum_B:.2f}", True, WHITE), (WIDTH - 180, 50))
    scoreboard = FONT.render(f"Score  A: {score_A}  |  B: {score_B}", True, WHITE)
    screen.blit(scoreboard, scoreboard.get_rect(center=(WIDTH // 2, 20)))
    pygame.display.flip()


if __name__ == "__main__":
    env = SoccerEnv()
    agent_A = HierarchicalPPOAgent(name="A")
    agent_B = HierarchicalPPOAgent(name="B")
    agent_A.policy.load_state_dict(torch.load("agent_A.pth"))
    agent_B.policy.load_state_dict(torch.load("agent_B.pth"))

    score_A = score_B = 0
    cum_r_A = cum_r_B = 0.0
    state = env.reset()
    start_time = time.time()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("[INFO] Manual reset triggered with R key.")
                state = env.reset()
                cum_r_A, cum_r_B = 0.0, 0.0
                start_time = time.time()

        macro_A, param_A = agent_A.select_action(state)
        macro_B, param_B = agent_B.select_action(state)

        # Immediate reward logic
        for macro, param, agent, label in [
            (macro_A, param_A, "A", 0),
            (macro_B, param_B, "B", 4),
        ]:
            if MACRO_ACTIONS[macro] == "shoot" and env.cooldowns[agent] == SHOOT_COOL_DOWN:
                reward = 50 if env.ball_owner == agent else -100
                if agent == "A":
                    cum_r_A += reward
                else:
                    cum_r_B += reward
            elif (
                MACRO_ACTIONS[macro] == "intercept"
                and env.cooldowns[agent] == INTERCEPT_COOL_DOWN
            ):
                agent_pos = state[label : label + 2]
                ball_pos = state[8:10]
                dist = np.linalg.norm(agent_pos - ball_pos)
                reward = -80 if env.ball_owner == agent else (-20 if dist >= 10 else 0)
                if reward:
                    if agent == "A":
                        cum_r_A += reward
                    else:
                        cum_r_B += reward

        state, r_A, r_B, done = env.step(macro_A, param_A, macro_B, param_B)

        if macro_A == 0:
            r_A += shaping_reward(state, "A")
        if macro_B == 0:
            r_B += shaping_reward(state, "B")

        cum_r_A += r_A
        cum_r_B += r_B

        if done:
            last_event = env.event_table[-1] if env.event_table else {}
            if last_event.get("type") == "goal":
                if last_event.get("agent") == "A":
                    score_A += 1
                elif last_event.get("agent") == "B":
                    score_B += 1
            state = env.reset()

        render(
            env,
            MACRO_ACTIONS[macro_A],
            MACRO_ACTIONS[macro_B],
            time.time() - start_time,
            cum_r_A,
            cum_r_B,
            score_A,
            score_B,
        )
        clock.tick(30)

    pygame.quit()
