import copy

class MinimaxAgent:
    def __init__(self, depth=2):
        self.depth = depth

    def select_move(self, goenv):
        legal_moves = goenv.get_legal_moves()
        if not legal_moves:
            return None

        best_move = None
        best_score = -float('inf')
        ai_color = goenv.get_active_player()

        # Minimax works by always using the AI's color for evaluation
        for move in legal_moves:
            temp_env = copy.deepcopy(goenv)
            #unpack move by *
            temp_env.play_move(*move)
            score = self.minimax_pruning(temp_env, ai_color, depth=self.depth-1, alpha=-float('inf'), beta=float('inf'), maximizing_player=False)
            
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def minimax_pruning(self, goenv, ai_color, depth, alpha, beta, maximizing_player):
        if depth == 0 or goenv.is_over():
            return self.evaluate(goenv, ai_color)

        current_player = goenv.get_active_player()
        legal_moves = goenv.get_legal_moves()

        if not legal_moves:
            goenv.pass_turn()
            return self.minimax_pruning(goenv, ai_color, depth-1, alpha, beta, not maximizing_player)

        if maximizing_player:
            max_eval = -float('inf') 
            for move in legal_moves:
                temp_env = copy.deepcopy(goenv)
                temp_env.play_move(*move)
                #opponent color
                evaluated = self.minimax_pruning(temp_env, ai_color, depth-1, alpha, beta, False)
                max_eval = max(max_eval, evaluated)
                alpha = max(alpha, evaluated)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                temp_env = copy.deepcopy(goenv)
                temp_env.play_move(*move)
                evaluated = self.minimax_pruning(temp_env, ai_color, depth-1, alpha, beta, True)
                min_eval = min(min_eval,evaluated)
                beta = min(beta, evaluated)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate(self, goenv, color):
        # Score = territory + captures + komi
        return goenv.score(color) - goenv.score(-color)