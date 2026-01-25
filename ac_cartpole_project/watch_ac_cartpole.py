import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# ==============================================================================
# THE NETWORK CLASS (Must match training script exactly)
# ==============================================================================
class ActorCritic(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(ActorCritic, self).__init__()
        self.affine1 = nn.Linear(n_observations, 128)
        self.action_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_prob, state_values

def watch():
    # Check if model file exists
    if not os.path.exists("ac_cartpole.pth"):
        print("Error: 'ac_cartpole.pth' not found. Please run the training script first.")
        return

    # Create Environment with visual rendering
    env = gym.make("CartPole-v1", render_mode="human")
    
    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n).to(device)
    
    # Load Weights
    model.load_state_dict(torch.load("ac_cartpole.pth", map_location=device))
    model.eval() # Set to evaluation mode
    print("Model loaded successfully!")
    
    print("Starting simulation... (Close window or press Ctrl+C to stop)")

    for i in range(5): # Play 5 episodes
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Prepare state for network
            state_tensor = torch.from_numpy(state).float().to(device)
            
            # Get action probabilities
            probs, val = model(state_tensor)
            
            # For demonstration, we pick the most likely action (argmax)
            # This shows the agent's "best" behavior
            action = torch.argmax(probs).item()
            
            # Take step
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        print(f"Episode {i+1}: Score {total_reward}")
    
    env.close()

if __name__ == "__main__":
    watch()