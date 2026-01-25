import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==============================================================================
# 1. THE NEURAL NETWORK 
# (MUST match the training script exactly, or it won't load!)
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
    # 1. Setup Environment with rendering (so you can see it)
    # 'render_mode="human"' opens the window
    env = gym.make("CartPole-v1", render_mode="human")
    
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # 2. Load the Neural Network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(n_observations, n_actions).to(device)
    
    # LOAD THE SAVED BRAIN
    try:
        model.load_state_dict(torch.load("dqn_cartpole_final.pth", map_location=device))
        model.eval() # Set to evaluation mode (turns off training specific layers)
        print("✓ Model loaded successfully!")
    except FileNotFoundError:
        print("✗ Error: 'dqn_cartpole_final.pth' not found.")
        print("  Make sure you ran the training script first!")
        return

    print("Starting simulation... (Press Ctrl+C to stop early)")

    # 3. Play 5 Games
    for i in range(5):
        state, _ = env.reset()
        # Convert state to tensor just like in training
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # ASK THE BRAIN: "What should I do?"
            with torch.no_grad():
                # We pick the action with the highest Q-value
                action = model(state).max(1)[1].view(1, 1)
            
            # DO THE ACTION
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Update state for next step
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        print(f"Game {i+1}: Survived {steps} steps (Score: {total_reward})")

    env.close()
    print("Demonstration finished.")

if __name__ == "__main__":
    watch() 