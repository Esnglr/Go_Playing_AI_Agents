# Go AI Project

A Python project for building AI agents to play the game of Go using reinforcement learning and classical AI techniques. This project includes:

- **Q-Learning Agent** – learns optimal moves through experience.  
- **DQN Agent** – uses a neural network to approximate Q-values.  
- **Minimax Agent** – classical depth-limited search agent.  
- **Negamax Agent** – 
- **Random Agent** -

---

## Features

- Train AI agents to play on a 5x5 Go board (customizable board size).  
- Supports Q-learning, DQN, and Minimax algorithms.  
- Track training performance using plots (`matplotlib`).  
- Save and load trained models (`pickle`).  
- Interactive play with AI agents.  

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

# Usage

