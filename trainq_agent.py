from go_environment import GoEnv
from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
from agents.negamax_agent import NegamaxAgent
import random


BOARD_SIZE = 5
MAX_EPISODES_SELFPLAY = 5000
MAX_EPISODES_RANDOM = 5000
MAX_EPISODES_MINIMAX = 7000
MAX_EPISODES_NEGAMAX = 7000
CHECKPOINT_EVERY = 1000
ROLLING_WINDOW = 100

REWARD_CAPTURE = 1.2
REWARD_LOSS = -0.8
REWARD_TERRITORY = 0.2

ALPHA = 0.15
GAMMA = 0.95 
EPSILON_START = 0.4
EPSILON_MIN = 0.1
EPSILON_DECAY_STEP = 0.002


def get_q_color(q_agent, black, white):
    if q_agent == black:
        return GoEnv.BLACK
    elif q_agent == white:
        return GoEnv.WHITE
    else:
        raise ValueError("Q agent not in game")


def play_game(agent_black, agent_white, q_agent=None, max_moves=50):
    game = GoEnv(size=BOARD_SIZE)
    moves_played = 0
    prev_captures = {GoEnv.BLACK: 0, GoEnv.WHITE: 0}

    while not game.is_over() and moves_played < max_moves:
        player = game.get_active_player()
        move = agent_black.select_move(game) if player == GoEnv.BLACK else agent_white.select_move(game)
        state = q_agent.get_state(game) if q_agent and player == get_q_color(q_agent, agent_black, agent_white) else None

        if move:
            before = dict(prev_captures)
            game.play_move(*move)
            next_state = q_agent.get_state(game) if state else None

            if state:
                captured_now = game.captures
                captured_enemy = captured_now[player] - before[player]
                lost_stones = captured_now[-player] - before[-player]
                territory_diff = game.count_territory(player) - game.count_territory(-player)

                reward = (REWARD_CAPTURE * captured_enemy) + (REWARD_LOSS * lost_stones) + (REWARD_TERRITORY * territory_diff)
                q_agent.update(state, move, reward, next_state, game.get_legal_moves())

            prev_captures = game.captures
        else:
            game.pass_turn()
            if state:
                next_state = q_agent.get_state(game)
                q_agent.update(state, None, -0.1, next_state, game.get_legal_moves())

        moves_played += 1

    black_score = game.score(GoEnv.BLACK)
    white_score = game.score(GoEnv.WHITE)
    winner = GoEnv.BLACK if black_score > white_score else GoEnv.WHITE if white_score > black_score else 0

    if q_agent:
        q_color = get_q_color(q_agent, agent_black, agent_white)
        final_reward = 1 if winner == q_color else -1 if winner != 0 else 0
        q_agent.update(state, None, final_reward, next_state, game.get_legal_moves())

    return winner



def train_against(q_agent, opponent_class, episodes, max_moves=50, epsilon_start=EPSILON_START):
    q_agent.epsilon = epsilon_start
    win_history = []

    for ep in range(1, episodes + 1):
        opponent = opponent_class()
        if random.random() < 0.5:
            black, white = q_agent, opponent
        else:
            black, white = opponent, q_agent

        winner = play_game(black, white, q_agent=q_agent, max_moves=max_moves)
        q_color = get_q_color(q_agent, black, white)
        win_history.append(1 if winner == q_color else 0)

        #keep rolling window
        if len(win_history) > ROLLING_WINDOW:
            win_history.pop(0)

        win_rate = sum(win_history) / len(win_history)

        if ep % ROLLING_WINDOW == 0 and q_agent.epsilon > EPSILON_MIN:
            q_agent.epsilon = max(EPSILON_MIN, q_agent.epsilon - EPSILON_DECAY_STEP)

        #print progress frequently
        if ep % 50 == 0:
            print(f"[{opponent_class.__name__}] Episode {ep}/{episodes} — RollingWinRate={win_rate:.2f} — ε={q_agent.epsilon:.3f}")

        #save checkpoint periodically
        if ep % CHECKPOINT_EVERY == 0:
            filename = f"checkpoint_vs_{opponent_class.__name__}_{ep}.pkl"
            q_agent.save(filename)
            print(f"Checkpoint saved: {filename}")

    #saving final Q-table
    filename = f"trained_q_vs_{opponent_class.__name__}2.pkl"
    q_agent.save(filename)
    print(f"✅ Finished training vs {opponent_class.__name__} — saved {filename}")



if __name__ == "__main__":
    q_agent = QLearningAgent(alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON_START)

    print("🔹 Starting training...")

    # Stage 1 — Self-play
    train_against(q_agent, QLearningAgent, MAX_EPISODES_SELFPLAY)
    #q_agent.load_multiple(["trained_q_vs_NegamaxAgent.pkl", "trained_q_vs_MinimaxAgent.pkl", "trained_q_vs_QLearningAgent.pkl","trained_q_vs_RandomAgent.pkl"])
    # Stage 2 — Random
    train_against(q_agent, RandomAgent, MAX_EPISODES_RANDOM)

    # Stage 3 — Minimax
    train_against(q_agent, MinimaxAgent, MAX_EPISODES_MINIMAX)

    # Stage 4 — Negamax
    train_against(q_agent, NegamaxAgent, MAX_EPISODES_NEGAMAX)