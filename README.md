# Go AI Project

A Python project for building AI agents to play the game of Go using reinforcement learning and classical AI techniques. This project includes:

- **Q-Learning Agent** – learns optimal moves through experience.  
- **DQN Agent** – uses a neural network to approximate Q-values.  
- **Minimax Agent** – classical depth-limited search agent.  
- **Negamax Agent** – variant of Minimax used in zero-sum games
- **Random Agent** - selects among legal moves randomly

---

## Features

- Train AI agents to play on a 5x5 Go board (customizable board size).  
- Supports Q-learning, DQN, and Minimax algorithms.  
- Track training performance using plots (`matplotlib`).  
- Save and load trained models (`pickle`).  
- Interactive play with AI agents.
- Benchmarking the win rates  

---

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/go-ai-project.git
cd go-ai-project
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

# Dependencies:

- numpy : for board representation and faster calculations
- torch : for Q-learning neural network and DQN
- mathplotlib : for plotting trainig results
- IPython (optional): for interactive debugging

# To train the Q-Learning and Deep Q-Learning agents:
```bash
python ./trainq_agent.py
python ./train_dqn_agent.py
```

# To play game with the agents:
```bash
python ./test_game.py
```

# To benchmark the agent's win rates:
```bash
python ./benchmark.py
```

---

## Observed Win Rates

After running all pairwise benchmarking matches, the following win statistics were observed:

| **Agent**     | **Win Count** | **Win Rate (%)** |
|----------------|---------------|------------------|
| Minimax        | 40            | 100.0            |
| Negamax        | 30            | 75.0             |
| Random         | 12            | 30.0             |
| Q-Learning     | 10            | 25.0             |
| DQN            | 8             | 20.0             |

The classical agents (Minimax and Negamax) played very well on the 5x5 board.  
The Q-Learning and DQN agents didn’t perform as well yet, but they could improve with more training and better hyperparameter tuning.  
Overall, both classical and learning-based methods showed their strengths in different ways.


