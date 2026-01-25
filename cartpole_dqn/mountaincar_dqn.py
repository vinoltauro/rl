import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque

# ==============================================================================
# 1. THE NEURAL NETWORK 
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

# ==============================================================================
# 2. HYPERPARAMETERS
# ==============================================================================
BATCH_SIZE = 64         
GAMMA = 0.99            
EPS_START = 1.0         
EPS_END = 0.01          
EPS_DECAY = 1000        # Faster decay because Hints help it learn faster!
TAU = 0.005             
LR = 1e-3               
EPISODES = 600          # Fewer episodes needed with hints

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 3. TRAINING FUNCTION WITH REWARD SHAPING
# ==============================================================================
def train_mountaincar_guided():
    env = gym.make("MountainCar-v0")
    n_actions = env.action_space.n
    n_observations = env.observation_space.shape[0]

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = deque(maxlen=20000) 

    steps_done = 0
    episode_rewards = []
    loss_history = []
    epsilon_history = []
    
    best_real_score = -float('inf')

    print(f"Training MountainCar (with Hints) on {device}...")
    
    for i_episode in range(EPISODES):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        current_episode_loss = []
        total_reward = 0
        real_score = 0 # Track the REAL score (without hints) to check success
        
        for t in range(2000): 
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            
            if sample > eps_threshold:
                with torch.no_grad():
                    action = policy_net(state).max(1)[1].view(1, 1)
            else:
                action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

            observation, reward, terminated, truncated, _ = env.step(action.item())
            
            # --- REWARD SHAPING (The Magic Fix) ---
            # Give a bonus for being high up on the hill
            position = observation[0]
            # 0.5 is the middle (bottom) of the valley
            # We reward distance from the bottom
            height_bonus = abs(position - (-0.5)) 
            
            # Custom Reward: -1 (time penalty) + height bonus
            shaped_reward = reward + (height_bonus * 10) 
            
            reward_tensor = torch.tensor([shaped_reward], device=device, dtype=torch.float32)
            done = terminated or truncated
            
            total_reward += shaped_reward
            real_score += reward # Keep track of the actual game score (-200 to -100)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.append((state, action, next_state, reward_tensor))
            state = next_state

            if len(memory) >= BATCH_SIZE:
                transitions = random.sample(memory, BATCH_SIZE)
                batch_state, batch_action, batch_next, batch_reward = zip(*transitions)
                state_batch = torch.cat(batch_state)
                action_batch = torch.cat(batch_action)
                reward_batch = torch.cat(batch_reward)
                state_action_values = policy_net(state_batch).gather(1, action_batch)
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next)), device=device, dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch_next if s is not None])
                next_state_values = torch.zeros(BATCH_SIZE, device=device)
                with torch.no_grad():
                    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
                expected_state_action_values = (next_state_values * GAMMA) + reward_batch
                criterion = nn.SmoothL1Loss()
                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
                current_episode_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                optimizer.step()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                break
        
        episode_rewards.append(real_score) # Plot the REAL score, not the shaped one
        epsilon_history.append(eps_threshold)
        loss_history.append(np.mean(current_episode_loss) if current_episode_loss else 0)

        # Save Best Logic (Using Real Score)
        avg_real = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else -200
        if len(episode_rewards) >= 50 and avg_real > best_real_score:
            best_real_score = avg_real
            torch.save(policy_net.state_dict(), "mountaincar_solved.pth")
            print(f"  --> Saved Best Model! Avg Real Score: {best_real_score:.1f}")

        if i_episode % 20 == 0:
            print(f"Ep {i_episode:3d} | Real Score: {real_score:6.0f} | Avg: {avg_real:6.1f} | Epsilon: {eps_threshold:.2f}")

    return policy_net, episode_rewards, loss_history, epsilon_history

# ==============================================================================
# 4. PLOTTING FUNCTION
# ==============================================================================
def plot_results(rewards, losses, epsilons):
    print("Generating final analysis...")
    fig = plt.figure(figsize=(20, 12))
    
    SOLVED_SCORE = -110

    plt.subplot(3, 3, 1)
    plt.plot(rewards, alpha=0.6, color='blue', linewidth=0.5)
    plt.title('Raw Game Scores (Higher is Better)', fontweight='bold')
    plt.axhline(y=SOLVED_SCORE, color='red', linestyle='--', label='Solved (-110)')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(3, 3, 2)
    window = 50
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(ma, color='green', linewidth=2)
    plt.title(f'Moving Average (window={window})', fontweight='bold')
    plt.axhline(y=SOLVED_SCORE, color='red', linestyle='--', label='Solved')
    plt.grid(alpha=0.3)

    plt.subplot(3, 3, 4)
    plt.plot(losses, color='purple', alpha=0.7)
    plt.yscale('log')
    plt.title('Training Loss', fontweight='bold')
    plt.grid(alpha=0.3)

    plt.subplot(3, 3, 5)
    plt.hist(rewards, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(x=SOLVED_SCORE, color='red', linestyle='--', label='Solved')
    plt.title('Score Distribution', fontweight='bold')
    plt.grid(alpha=0.3)

    plt.subplot(3, 3, 9)
    plt.axis('off')
    txt = f"""
    MOUNTAIN CAR (SHAPED)
    =====================
    Episodes: {len(rewards)}
    Best 50-avg: {np.max(np.convolve(rewards, np.ones(50)/50, mode='valid')):.1f}
    """
    plt.text(0.1, 0.3, txt, fontsize=12, family='monospace')

    plt.tight_layout()
    plt.savefig('mountaincar_shaped_analysis.png', dpi=200)
    print("Graph saved as 'mountaincar_shaped_analysis.png'")
    plt.show()

if __name__ == "__main__":
    model, rewards, losses, epsilons = train_mountaincar_guided()
    plot_results(rewards, losses, epsilons)