import random
import pickle

class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.Q = {} # Q[state][action] = Q
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state(self, goenv):
        return tuple(tuple(row) for row in goenv.board)

    def select_move(self, goenv):
        legal_moves = goenv.get_legal_moves()
        if not legal_moves:
            return None

        state = self.get_state(goenv)
        if state not in self.Q:
            self.Q[state] = {move: 0.0 for move in legal_moves}

        q_values = self.Q[state]
        if not q_values:
            return None

        # Epsilon-greedy policy : stochastic policy
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        else:
            q_values = self.Q[state]
            max_q = max(q_values.values())
            best_moves = [move for move, value in q_values.items() if value == max_q]
            return random.choice(best_moves)

    def update(self, state, action, reward, next_state, next_legal_moves):
        if state not in self.Q:
            self.Q[state] = {action: 0.0}
        if next_state not in self.Q:
            self.Q[next_state] = {move: 0.0 for move in next_legal_moves}

        old_q = self.Q[state].get(action, 0.0)
        max_next_q = max(self.Q[next_state].values()) if self.Q[next_state] else 0.0

        # Standard Q-learning Bellman update rule
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.Q[state][action] = new_q

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.Q, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.Q = pickle.load(f)

    def load_multiple(self, filenames):

        import pickle
        for fname in filenames:
            with open(fname, "rb") as f:
                loaded_Q = pickle.load(f)
                for state, moves in loaded_Q.items():
                    if state not in self.Q:
                        self.Q[state] = moves
                    else:
                        # Merge moves: overwrite only missing ones
                        for move, value in moves.items():
                            if move not in self.Q[state]:
                                self.Q[state][move] = value