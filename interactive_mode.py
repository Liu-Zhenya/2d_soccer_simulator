# interactive_mode.py
import pygame
import numpy as np
from env.simulator import SoccerEnv
import math
import time
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
    if norm >= 1e-6:
        unit = direction / norm
        draw_arrow.last_dirs[agent_id] = unit
    else:
        unit = draw_arrow.last_dirs[agent_id]

    end = (start[0] + length * unit[0], start[1] + length * unit[1])
    pygame.draw.line(screen, color, start, end, 3)


def shaping_reward(state, agent):
    idx = 0 if agent == "A" else 4
    agent_pos = state[idx : idx + 2]
    ball_pos = state[8:10]

    ball_dist = np.linalg.norm(agent_pos - ball_pos)
    # return 0.5 if ball_dist <= 3.0 else 0.9 * (1 / ball_dist)
    return 0.5 if ball_dist <= 3.0 else 0


def render(
    env,
    macro_str_A,
    macro_str_B,
    elapsed_time,
    cumulative_reward_A,
    cumulative_reward_B,
    score_A,
    score_B,
):
    goal_height = int(14 * SCALE[1])
    screen.fill(GREEN)
    pygame.draw.rect(screen, WHITE, pygame.Rect(0, 0, WIDTH, HEIGHT), 5)
    pygame.draw.line(screen, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 2)
    pygame.draw.circle(screen, WHITE, (WIDTH // 2, HEIGHT // 2), int(10 * SCALE[0]), 2)

    box_width = int(12 * SCALE[0])
    box_height = int(24 * SCALE[1])
    pygame.draw.rect(
        screen, WHITE, pygame.Rect(0, (HEIGHT - box_height) // 2, box_width, box_height), 2
    )
    pygame.draw.rect(
        screen,
        WHITE,
        pygame.Rect(WIDTH - box_width, (HEIGHT - box_height) // 2, box_width, box_height),
        2,
    )

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

    pos_A = to_px(env.state[0:2])
    vel_A = env.state[2:4]
    pygame.draw.circle(screen, BLUE, pos_A, 20)
    draw_arrow(pos_A, vel_A, agent_id="A", color=YELLOW)

    pos_B = to_px(env.state[4:6])
    vel_B = env.state[6:8]
    pygame.draw.circle(screen, RED, pos_B, 20)
    draw_arrow(pos_B, vel_B, agent_id="B", color=YELLOW)

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
    screen.blit(
        FONT.render(f"Reward A: {cumulative_reward_A:.2f}", True, WHITE), (WIDTH - 180, 30)
    )
    screen.blit(
        FONT.render(f"Reward B: {cumulative_reward_B:.2f}", True, WHITE), (WIDTH - 180, 50)
    )

    scoreboard = FONT.render(f"Score  A: {score_A}  |  B: {score_B}", True, WHITE)
    scoreboard_rect = scoreboard.get_rect(center=(WIDTH // 2, 20))
    screen.blit(scoreboard, scoreboard_rect)

    pygame.display.flip()


def get_direction(keys, key_map):
    dx = dy = 0
    if keys[key_map["left"]]:
        dx -= 1
    if keys[key_map["right"]]:
        dx += 1
    if keys[key_map["up"]]:
        dy -= 1
    if keys[key_map["down"]]:
        dy += 1
    if dx == 0 and dy == 0:
        return None
    return math.atan2(dy, dx)


def main():
    env = SoccerEnv()
    env.reset()
    running = True

    buffer_A = ReplayBuffer()
    buffer_B = ReplayBuffer()
    macro_history_A = []
    macro_history_B = []

    angle_A = angle_B = 0.0
    score_A = score_B = 0
    cumulative_reward_A = cumulative_reward_B = 0.0
    start_time = time.time()

    key_map_A = {
        "left": pygame.K_LEFT,
        "right": pygame.K_RIGHT,
        "up": pygame.K_UP,
        "down": pygame.K_DOWN,
    }
    key_map_B = {
        "left": pygame.K_a,
        "right": pygame.K_d,
        "up": pygame.K_w,
        "down": pygame.K_s,
    }

    print("Arrow keys to move A | Space=shoot | J=intercept")
    print("WASD to move B | Enter=shoot | K=intercept")

    while running:
        keys = pygame.key.get_pressed()
        dir_angle_A = get_direction(keys, key_map_A)
        dir_angle_B = get_direction(keys, key_map_B)
        if dir_angle_A is not None:
            angle_A = dir_angle_A
        if dir_angle_B is not None:
            angle_B = dir_angle_B

        shoot_A = keys[pygame.K_SPACE]
        intercept_A = keys[pygame.K_j]
        shoot_B = keys[pygame.K_RETURN]
        intercept_B = keys[pygame.K_k]

        macro_A, power_A = (
            (1, 1.0)
            if shoot_A
            else (
                (2, 1.0)
                if intercept_A
                else (0, 1.0) if dir_angle_A is not None else (0, 0.0)
            )
        )
        macro_B, power_B = (
            (1, 1.0)
            if shoot_B
            else (
                (2, 1.0)
                if intercept_B
                else (0, 1.0) if dir_angle_B is not None else (0, 0.0)
            )
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False

        param_A = np.array([angle_A, power_A])
        param_B = np.array([angle_B, power_B])

        # Immediate reward logic before step
        if macro_A != -1:
            if MACRO_ACTIONS[macro_A] == "shoot" and env.cooldowns["A"] == SHOOT_COOL_DOWN:

                reward = 50 if env.ball_owner == "A" else -100
                buffer_A.add(
                    (env.state.copy(), macro_A, param_A, reward, env.state.copy(), False)
                )
                print(
                    f"[A] buffer added: action={['move','shoot','intercept'][1]}, immediate reward={reward}"
                )
                cumulative_reward_A += reward
            elif (
                MACRO_ACTIONS[macro_A] == "intercept"
                and env.cooldowns["A"] == INTERCEPT_COOL_DOWN
            ):
                agent_pos = env.state[0:2]
                ball_pos = env.state[8:10]
                dist = np.linalg.norm(agent_pos - ball_pos)
                reward = -80 if env.ball_owner == "A" else (-20 if dist >= 10 else 0)
                if reward != 0:
                    buffer_A.add(
                        (
                            env.state.copy(),
                            macro_A,
                            param_A,
                            reward,
                            env.state.copy(),
                            False,
                        )
                    )
                    cumulative_reward_A += reward

            # macro_history_A.append((env.state.copy(), macro_A, param_A))

        if macro_B != -1:
            if MACRO_ACTIONS[macro_B] == "shoot" and env.cooldowns["B"] == SHOOT_COOL_DOWN:
                reward = 50 if env.ball_owner == "B" else -100
                buffer_B.add(
                    (env.state.copy(), macro_B, param_B, reward, env.state.copy(), False)
                )
                cumulative_reward_B += reward
            elif (
                MACRO_ACTIONS[macro_B] == "intercept"
                and env.cooldowns["B"] == INTERCEPT_COOL_DOWN
            ):
                agent_pos = env.state[4:6]
                ball_pos = env.state[8:10]
                dist = np.linalg.norm(agent_pos - ball_pos)
                reward = -80 if env.ball_owner == "B" else (-20 if dist >= 10 else 0)
                if reward != 0:
                    buffer_B.add(
                        (
                            env.state.copy(),
                            macro_B,
                            param_B,
                            reward,
                            env.state.copy(),
                            False,
                        )
                    )
                    cumulative_reward_B += reward

            # macro_history_B.append((env.state.copy(), macro_B, param_B))

        if env.cooldowns["A"] == 0 and macro_A in [1, 2]:
            macro_history_A.append((env.state.copy(), macro_A, param_A))
        if env.cooldowns["B"] == 0 and macro_B in [1, 2]:
            macro_history_B.append((env.state.copy(), macro_B, param_B))

        _, r_A, r_B, done = env.step(macro_A, param_A, macro_B, param_B)

        # Apply shaping reward
        if macro_A == 0:
            r_A += shaping_reward(env.state, "A")
        if macro_B == 0:
            r_B += shaping_reward(env.state, "B")

        cumulative_reward_A += r_A
        cumulative_reward_B += r_B

        if macro_history_A and r_A != 0:
            s, m, p = macro_history_A.pop(0)
            buffer_A.add((s, m, p, r_A, env.state.copy(), done))
            print(
                f"[A] buffer added: action={['move','shoot','intercept'][m]}, param={p}, reward={r_A:.4f}, done={done}"
            )
        if macro_history_B and r_B != 0:
            s, m, p = macro_history_B.pop(0)
            buffer_A.add((s, m, p, r_B, env.state.copy(), done))
            print(
                f"[B] buffer added: action={['move','shoot','intercept'][m]}, param={p}, reward={r_B:.4f}, done={done}"
            )

        if done:
            last_event = env.event_table[-1] if env.event_table else {}
            print(f"Event: {last_event}")
            if last_event.get("type") == "goal":
                if last_event.get("agent") == "A":
                    score_A += 1
                    # cumulative_reward_A += 100
                elif last_event.get("agent") == "B":
                    score_B += 1
                    # cumulative_reward_B += 100
                env.reset()
            elif last_event.get("type") == "sideline_reset":
                env.done = False
            else:
                env.reset()

        elapsed_time = time.time() - start_time
        render(
            env,
            ["move", "shoot", "intercept"][macro_A],
            ["move", "shoot", "intercept"][macro_B],
            elapsed_time,
            cumulative_reward_A,
            cumulative_reward_B,
            score_A,
            score_B,
        )
        clock.tick(30)


if __name__ == "__main__":
    main()
