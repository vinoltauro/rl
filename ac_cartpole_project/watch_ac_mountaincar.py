import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

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

def watch():
    env = gym.make("MountainCar-v0", render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    
    if os.path.exists("dqn_mountaincar_nuclear.pth"):
        model.load_state_dict(torch.load("dqn_mountaincar_nuclear.pth", map_location=device))
        print("Loaded Nuclear Model!")
    else:
        print("Model not found!")
        return

    for i in range(5):
        state, _ = env.reset()
        done = False
        steps = 0
        while not done:
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = model(state_t).max(1)[1].view(1, 1)
            state, _, done, truncated, _ = env.step(action.item())
            steps += 1
            if done or truncated: break
        print(f"Game {i+1}: Finished in {steps} steps")
    
    env.close()

if __name__ == "__main__":
    watch()