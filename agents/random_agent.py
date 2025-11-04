import random

class RandomAgent:

    def select_move(self, goenv):
        legal_moves = goenv.get_legal_moves()
        if not legal_moves:
            return None
        return random.choice(legal_moves)