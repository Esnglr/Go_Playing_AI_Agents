import copy

class NegamaxAgent:
    def __init__(self, depth=2):
        self.depth = depth

    def select_move(self, goenv):
        #current_player always be AI itself
        best_move , _ = self.negamax(goenv, self.depth, goenv.current_player)
        if best_move is None:
            goenv.pass_turn()
        return best_move

    def negamax(self, goenv, depth, color, alpha=-float('inf'), beta=float('inf')):
        if depth == 0 or goenv.is_over():
            return None, self.evaluate(goenv, color) 
        
        best_score = -float('inf')
        best_move = None

        for move in goenv.get_legal_moves():
            temp_env = copy.deepcopy(goenv)
            temp_env.play_move(*move)
            
            _, score = self.negamax(temp_env,depth -1, -color, -beta, -alpha)
            score = -score

            if score > best_score:
                best_score = score
                best_move = move

            #alpha-beta pruning
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return best_move, best_score


    # everytime, in the eyes of AI 
    def evaluate(self, goenv, color):
        return goenv.score(color) - goenv.score(-color)