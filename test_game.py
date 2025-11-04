from go_environment import GoEnv
from agents.minimax_agent import MinimaxAgent
from agents.negamax_agent import NegamaxAgent
from agents.random_agent import RandomAgent
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent, device
import torch
import random
import time

def parse_move(move_str, board_size):
    move_str = move_str.strip().upper()
    if move_str == "PASS":
        return "pass"

    if len(move_str) < 2:
        return None
    
    col_char = move_str[0]
    row_str = move_str[1:]

    # Skip 'I' column if board_size >= 9
    cols = [chr(ord('A') + i) for i in range(board_size + 1)]
    if 'I' in cols:
        cols.remove('I')

    if col_char not in cols:
        return None

    try:
        row = int(row_str) - 1
    except ValueError:
        return None

    x = cols.index(col_char)
    y = row
    if x >= board_size or y >= board_size:
        return None

    return (x, y)

def setup_game(board_size=5, agent_type="negamax", depth=2, q_model=None, device="cpu"):
    game = GoEnv(size=board_size)
    agent_type = agent_type.lower()

    if agent_type == "minimax":
        agent = MinimaxAgent(depth=depth)
    elif agent_type == "negamax":
        agent = NegamaxAgent(depth=depth)
    elif agent_type == "random":
        agent = RandomAgent()
    elif agent_type == "qlearning":
        agent = QLearningAgent()
        if q_model:
            agent.load_multiple(q_model)
    elif agent_type == "dqn":
        agent = DQNAgent(n_observations=board_size*board_size, n_actions=board_size*board_size+1,memory_capacity=1000,lr=0.001,batch_size=1,gamma=0.99,eps_start=0.05, eps_end=0.05,eps_decay=1,tau=0.005)
        if q_model:
            agent.load(q_model)
    else:
        raise ValueError("Invalid agent type.")

    human_color = random.choice([GoEnv.BLACK, GoEnv.WHITE])
    ai_color = -human_color

    print(f"\n--- NEW GAME ---")
    print(f"Human: {'BLACK' if human_color == GoEnv.BLACK else 'WHITE'}")
    print(f"AI ({agent_type.upper()}): {'BLACK' if ai_color == GoEnv.BLACK else 'WHITE'}")
    print("Board size:", board_size)
    print("--------------------\n")

    return game, agent, human_color, ai_color


def play_game(game, agent, human_color, ai_color):
    while not game.is_over():
        print(game)
        current_player = game.get_active_player()

        if current_player == human_color:
            move_input = input("Enter move (e.g., A5) or 'pass': ").strip()
            parsed = parse_move(move_input, game.size)
            
            if parsed == "pass":
                game.pass_turn()
                print("You passed.")
                continue
            elif parsed is None:
                print("Invalid move! Try again.")
                continue

            x, y = parsed
            if not game.play_move(x,y):
                print("Illegal move! Try again.")
                continue

            print(f"You played at {move_input.upper()}")

        # AI
        else:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                game.pass_turn()
                print("AI passes.")
                continue

            print("AI is thinking...")
            start = time.time()
            ai_move = agent.select_move(game)
            duration = time.time() - start

            if ai_move is None:
                game.pass_turn()
                print("AI passes.\n")
            else:
                x,y = ai_move
                game.play_move(x,y)
                col_letter = chr(ord('A') + x + (1 if x >= 8 else 0))  # skip 'I'
                print(f"AI plays at {col_letter}{y + 1} (took {duration:.2f}s)\n")


    
    print("\nGame over!")
    print(game)
    black_score = game.score(GoEnv.BLACK)
    white_score = game.score(GoEnv.WHITE)
    print(f"Final score — BLACK: {black_score:.1f}, WHITE: {white_score:.1f}")
    if black_score > white_score:
        print("Winner: BLACK")
    elif white_score > black_score:
        print("Winner: WHITE")
    else:
        print("Result: TIE")


if __name__ == "__main__":
    print("Welcome to Go!\n")
    print("Choose AI agent:")
    print("1 - Minimax")
    print("2 - Negamax")
    print("3 - Random")
    print("4 - Q-Learning")
    print("5 - DQN")

    choice = input("Select one number for agent:").strip()
    
    AGENT_TYPE = {"1": "minimax", "2": "negamax", "3": "random", "4": "qlearning", "5": "dqn"}.get(choice, "negamax")
    BOARD_SIZE = 5
    DEPTH = 3
    Q_MODEL_PATH_LIST = ["pkl_files/trained_q_vs_MinimaxAgent2.pkl"]
    DQN_MODEL_PATH = "final/dqn_policy_q_values.pth"

    device_choice = "cuda" if torch.cuda.is_available() else "cpu"
    
    if choice == "4":
        game, agent, human_color, ai_color = setup_game(BOARD_SIZE, AGENT_TYPE, DEPTH, q_model=Q_MODEL_PATH_LIST)
    elif choice == "5":
        game, agent, human_color, ai_color = setup_game(BOARD_SIZE, AGENT_TYPE, DEPTH, q_model=DQN_MODEL_PATH, device=device_choice)
    else:
        game, agent, human_color, ai_color = setup_game(BOARD_SIZE, AGENT_TYPE, DEPTH)
    
    play_game(game, agent, human_color, ai_color)