import math
import matplotlib.pyplot as plt
#from itertools import count
import torch
import sys
from IPython import display
import os
from agents.minimax_agent import MinimaxAgent
from go_environment import GoEnv
from agents.dqn_agent import DQNAgent, device

num_episodes_phase1 = 20000  # self-play, high exploration
num_episodes_phase2 = 20000   # self-play, low exploration
num_episodes_phase3 = 200   # vs heuristic opponent

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

env = GoEnv(size=5)
n_observations = env.size * env.size
n_actions = env.size * env.size + 1
MAX_MOVES = 75 

agent = DQNAgent(n_observations, n_actions,
                 memory_capacity=50000,
                 batch_size=128,
                 gamma=0.99,
                 lr=3e-4,
                 tau=0.005)

episode_durations = []
is_ipython = 'ipykernel' in sys.modules

# Load existing checkpoint if any
#checkpoint_path = os.path.join(checkpoint_dir, "dqn_policy_q_values.pth")
#if os.path.exists(checkpoint_path):
#    agent.load(checkpoint_path)

def plot_durations():
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.clf()
    plt.title('Training')
    plt.xlabel('Episode')
    plt.ylabel('Moves')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)
    if is_ipython:
        display.display(plt.gcf())
        display.clear_output(wait=True)

def train(agent, num_episodes, eps_start, eps_end, eps_decay, opponent=None):
    agent.eps_start = eps_start
    agent.eps_end = eps_end
    agent.eps_decay = eps_decay

    for i_episode in range(num_episodes):
        env.reset()
        state = torch.tensor(env.board.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
        prev_captures = env.captures.copy()
        move_count = 0

        while not env.is_over() and move_count < MAX_MOVES:
            current_player = env.current_player

            if opponent is None or current_player == agent_color:
                action_tensor = agent.select_action(state, env)
                if action_tensor is not None:
                    action_index = int(action_tensor.item())
                    if action_index != env.size * env.size:
                        y, x = divmod(action_index, env.size)
                        valid_move = env.play_move(x, y)
                    else:
                        env.pass_turn()
                        valid_move = True
                else:
                    env.pass_turn()
                    valid_move = True
            else:
                move = opponent.select_move(env)
                if move is not None:
                    valid_move = env.play_move(*move)
                else:
                    env.pass_turn()
                    valid_move = True

            reward_value = 0.0
            if current_player == agent_color:
                if not valid_move:
                    reward_value = -0.5
                else:
                    captured_now = env.captures
                    captured_enemy = captured_now[current_player] - prev_captures[current_player]
                    lost_stones = captured_now[-current_player] - prev_captures[-current_player]
                    territory_diff = env.count_territory(current_player) - env.count_territory(-current_player)
                    reward_value = 1.0 * captured_enemy - 1.0 * lost_stones + 0.1 * territory_diff + 0.1
                    prev_captures = captured_now.copy()

                if env.is_over():
                    black_score = env.score(env.BLACK)
                    white_score = env.score(env.WHITE)
                    winner = env.BLACK if black_score > white_score else env.WHITE if white_score > black_score else 0
                    if winner == current_player:
                        reward_value += 1.0
                    elif winner == 0:
                        reward_value += 0.0
                    else:
                        reward_value += -1.0

                reward = torch.tensor([reward_value], dtype=torch.float32, device=device)
                next_state = None if env.is_over() else torch.tensor(env.board.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
                agent.memory.push(state, action_tensor, next_state, reward)
                agent.optimize_model()
                agent.soft_update_target()
                state = next_state

            move_count += 1

        episode_durations.append(move_count)
        print(f"Episode {i_episode} completed")
        if (i_episode + 1) % 100 == 0:
            print(f"Episode {i_episode + 1}/{num_episodes}, moves: {move_count}")
            plot_durations()

    agent.save("final/dqn_policy_q_values.pth")
    plot_durations()

# Phase 1: Self-play, high exploration
print("Phase 1: Self-play, high epsilon")
agent_color = env.BLACK
train(agent, num_episodes_phase1, eps_start=0.9, eps_end=0.5, eps_decay=5000)

# Phase 2: Self-play, low exploration
print("Phase 2: Self-play, low epsilon")
train(agent, num_episodes_phase2, eps_start=0.5, eps_end=0.05, eps_decay=2500)

# Phase 3: vs heuristic opponent (Negamax/Minimax)
print("Phase 3: vs heuristic opponent")
opponent = MinimaxAgent(depth=4)
train(agent, num_episodes_phase3, eps_start=0.05, eps_end=0.01, eps_decay=1000, opponent=opponent)

print("Training complete!")