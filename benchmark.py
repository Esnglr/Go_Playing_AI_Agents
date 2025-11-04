import itertools
import os
import torch
from collections import defaultdict

class GoBenchmark:

    def __init__(self, agents, n_games=10):
        self.agents = agents
        self.n_games = n_games
        self.win_counts = defaultdict(int)

    def play_game(self, agent_a_obj, agent_b_obj, go_env_class):
        env = go_env_class()
        env.reset()
        agents_map = {env.BLACK: agent_a_obj, env.WHITE: agent_b_obj}

        move_count = 0
        max_moves = env.size * env.size * 2

        while not env.is_over() and move_count < max_moves:
            current_player = env.get_active_player()
            agent = agents_map[current_player]
            move = agent.select_move(env) if hasattr(agent, "select_move") else None
            if move is None:
                env.pass_turn()
            else:
                env.play_move(move[0], move[1])
            move_count += 1

        # --- Scoring ---
        black_score = env.score(env.BLACK)
        white_score = env.score(env.WHITE)

        if black_score > white_score:
            winner_name = agent_a_obj.__class__.__name__
        elif white_score > black_score:
            winner_name = agent_b_obj.__class__.__name__
        else:
            winner_name = "Draw"

        return winner_name

    def run_round_robin(self, go_env_class):
        agent_names = list(self.agents.keys())

        for a_name, b_name in itertools.combinations(agent_names, 2):
            agent_a = self.agents[a_name]
            agent_b = self.agents[b_name]

            # --- Auto-load pretrained models if available ---
            if isinstance(agent_a, QLearningAgent):
                path_a = "pkl_files/trained_q_vs_MinimaxAgent2.pkl"
                if hasattr(agent_a, "load") and os.path.exists(path_a):
                    agent_a.load(path_a)
                    print(f"Loaded Q-learning model for {a_name}")
            elif isinstance(agent_a, DQNAgent):
                path_a = "checkpoints/dqn_policy_q_values.pth"
                if hasattr(agent_a, "load") and os.path.exists(path_a):
                    agent_a.load(path_a)
                    print(f"Loaded DQN model for {a_name}")

            if isinstance(agent_b, QLearningAgent):
                path_b = "pkl_files/trained_q_vs_MinimaxAgent2.pkl"
                if hasattr(agent_b, "load") and os.path.exists(path_b):
                    agent_b.load(path_b)
                    print(f"Loaded Q-learning model for {b_name}")
            elif isinstance(agent_b, DQNAgent):
                path_b = "checkpoints/dqn_policy_q_values.pth"
                if hasattr(agent_b, "load") and os.path.exists(path_b):
                    agent_b.load(path_b)
                    print(f"Loaded DQN model for {b_name}")

            # --- Play multiple games ---
            for i in range(self.n_games):
                winner = self.play_game(agent_a, agent_b, go_env_class)
                print(f"{a_name} vs {b_name}, Game {i+1}/{self.n_games} done. Winner: {winner}")

                if winner != "Draw":
                    self.win_counts[winner] += 1

        print("\n=== Final Win Counts ===")
        for name, wins in self.win_counts.items():
            print(f"{name}: {wins} wins")

if __name__ == "__main__":
    from go_environment import GoEnv
    from agents.random_agent import RandomAgent
    from agents.minimax_agent import MinimaxAgent
    from agents.negamax_agent import NegamaxAgent
    from agents.q_learning_agent import QLearningAgent
    from agents.dqn_agent import DQNAgent

    agents = {
        "Random": RandomAgent(),
        "Minimax": MinimaxAgent(depth=4),
        "Negamax": NegamaxAgent(depth=4),
        "Q-Learning": QLearningAgent(),
        "DQN": DQNAgent(n_observations=25, n_actions=25)
    }

    benchmark = GoBenchmark(agents, n_games=10)
    benchmark.run_round_robin(GoEnv)