import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==============================================================================
# 1. THE NEURAL NETWORK 
# (Must match the training script exactly)
# ==============================================================================
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
    # 1. Setup Environment with rendering
    # render_mode="human" makes the window pop up
    env = gym.make("MountainCar-v0", render_mode="human")
    
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # 2. Load the Neural Network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(n_observations, n_actions).to(device)
    
    try:
        # Load the weights we just trained
        model.load_state_dict(torch.load("mountaincar_solved.pth", map_location=device))
        model.eval() # Set to evaluation mode
        print("✓ MountainCar model loaded!")
    except FileNotFoundError:
        print("✗ Error: 'mountaincar_solved.pth' not found.")
        print("  Make sure you ran the training script first!")
        return

    print("Starting simulation... (Press Ctrl+C to stop)")

    # 3. Play 5 Games
    for i in range(5):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Slow down slightly so we can see it better (optional)
            # import time; time.sleep(0.01)

            with torch.no_grad():
                action = model(state).max(1)[1].view(1, 1)
            
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # In MountainCar, a score closer to -100 is great. -200 means it failed.
        print(f"Game {i+1}: Finished in {steps} steps (Score: {total_reward:.0f})")

    env.close()
    print("Demonstration finished.")

if __name__ == "__main__":
    watch()