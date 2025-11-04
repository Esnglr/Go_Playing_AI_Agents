import math
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent:
    def __init__(self, n_observations, n_actions, memory_capacity=10000, lr=3e-4,
                 batch_size=128, gamma=0.99, eps_start=0.9, eps_end=0.01, eps_decay=2500, tau=0.005):
        self.n_actions = n_actions
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(memory_capacity)

        self.steps_done = 0
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau

    def select_action(self, state, env):
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            action_index = env.size * env.size
            return torch.tensor([[action_index]], device=device, dtype=torch.long)

        legal_indices = [y * env.size + x for (x, y) in legal_moves]

        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        else:
            state_tensor = state.to(device)

        if sample > eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).squeeze(0)
                mask = q_values.clone()
                illegal_indices = [i for i in range(self.n_actions) if i not in legal_indices]
                mask[illegal_indices] = -float('inf')
                #mask = torch.full((self.n_actions,), -float('inf'), device=device)
                #mask[legal_indices] = q_values[legal_indices]
                action_index = mask.argmax().view(1, 1)
                return action_index
        else:
            action_index = random.choice(legal_indices)
            return torch.tensor([[action_index]], device=device, dtype=torch.long)


    # because i got lots of infinite loops, i need serious checking
    # also better for handling test game
    def select_move(self, env):
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            return None

        state = torch.tensor(env.board.flatten(), dtype=torch.float32, device=device)
        action_tensor = self.select_action(state, env)
        index = action_tensor.item()
        x, y = divmod(index, env.size)

        if (x,y) not in legal_moves:
            x,y = random.choice(legal_moves)

        return (x,y)


    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def soft_update_target(self):
        target_state_dict = self.target_net.state_dict()
        policy_state_dict = self.policy_net.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key] * self.tau + target_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_state_dict)

    def save(self, path="dqn_policy.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)
        print(f"Policy network saved at: {path}")

    def load(self, path="dqn_policy.pth"):
        
        print(f"🔍 Loading from: {os.path.abspath(path)}")
        self.policy_net.load_state_dict(torch.load(path, map_location=device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print(f"Policy network loaded from: {path}")
