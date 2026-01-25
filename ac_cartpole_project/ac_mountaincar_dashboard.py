import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import os

# ==============================================================================
# 1. THE BRAIN (Standard DQN)
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
# 2. CONFIGURATION
# ==============================================================================
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 1000  
TARGET_UPDATE = 10
LEARNING_RATE = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 3. TRAINING LOOP
# ==============================================================================
def train_nuclear_dashboard():
    # Extend time limit slightly to allow discovery
    env = gym.make("MountainCar-v0", max_episode_steps=400)
    
    n_actions = env.action_space.n
    n_observations = env.observation_space.shape[0]
    
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = deque(maxlen=20000)
    
    steps_done = 0
    
    # Stats for dashboard
    episode_rewards = [] # Real scores
    loss_history = []
    epsilon_history = []
    
    print(f"Nuclear Option Training on {device}...")

    for i_episode in range(600): # 600 Episodes
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        total_reward = 0
        episode_losses = []
        
        for t in range(400): # Max steps
            # Epsilon Greedy
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                np.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            
            if sample > eps_threshold:
                with torch.no_grad():
                    action = policy_net(state).max(1)[1].view(1, 1)
            else:
                action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

            next_state_np, reward, terminated, truncated, _ = env.step(action.item())
            
            # --- SUPER REWARD SHAPING (FIXED) ---
            pos = next_state_np[0]
            
            # THE FIX: IGNORE THE -1 REWARD. ONLY FOCUS ON HEIGHT.
            # pos is between -1.2 and 0.6.
            # We want to encourage getting away from -0.5 (bottom).
            height_bonus = abs(pos + 0.5) 
            
            # We use ONLY the bonus as the reward.
            # This guarantees POSITIVE feedback for climbing.
            shaped_reward = height_bonus * 10.0 
            
            # HUGE BONUS for hitting flag
            if pos >= 0.5:
                shaped_reward += 100.0
            
            total_reward += reward # Track REAL reward (-1 per step)
            
            shaped_reward_t = torch.tensor([shaped_reward], device=device, dtype=torch.float32)
            
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(next_state_np, dtype=torch.float32, device=device).unsqueeze(0)

            memory.append((state, action, next_state, shaped_reward_t))
            state = next_state

            # Optimize
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
                
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
                episode_losses.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if terminated or truncated:
                break
        
        episode_rewards.append(total_reward)
        epsilon_history.append(eps_threshold)
        loss_history.append(np.mean(episode_losses) if episode_losses else 0)
        
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if i_episode % 20 == 0:
            print(f"Ep {i_episode} | Real Score: {total_reward:.1f} | Loss: {loss_history[-1]:.3f}")

    torch.save(policy_net.state_dict(), "dqn_mountaincar_nuclear.pth")
    return episode_rewards, loss_history, epsilon_history

# ==============================================================================
# 4. DASHBOARD PLOTTING
# ==============================================================================
def plot_dashboard(rewards, losses, epsilons):
    print("Generating comprehensive analysis...")
    fig = plt.figure(figsize=(20, 12))
    
    SOLVED_SCORE = -110

    # 1. Raw Rewards
    plt.subplot(3, 3, 1)
    plt.plot(rewards, alpha=0.6, color='blue', linewidth=0.5)
    plt.title('Raw Rewards (Real Score)', fontweight='bold')
    plt.axhline(y=SOLVED_SCORE, color='green', linestyle='--', label='Solved (-110)')
    plt.legend()
    plt.grid(alpha=0.3)

    # 2. Moving Average
    plt.subplot(3, 3, 2)
    window = 50
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(ma, color='green', linewidth=2)
    plt.title(f'Moving Average (window={window})', fontweight='bold')
    plt.axhline(y=SOLVED_SCORE, color='green', linestyle='--', label='Solved')
    plt.grid(alpha=0.3)

    # 3. Epsilon Decay
    plt.subplot(3, 3, 3)
    plt.plot(epsilons, color='orange', linewidth=2)
    plt.title('Epsilon Decay', fontweight='bold')
    plt.grid(alpha=0.3)

    # 4. Loss
    plt.subplot(3, 3, 4)
    plt.plot(losses, color='purple', alpha=0.6)
    plt.title('Training Loss', fontweight='bold')
    plt.yscale('log')
    plt.grid(alpha=0.3)

    # 5. Reward Dist
    plt.subplot(3, 3, 5)
    plt.hist(rewards, bins=30, color='skyblue', edgecolor='black')
    plt.title('Score Distribution', fontweight='bold')
    plt.grid(alpha=0.3)

    # 6. Success Rate
    plt.subplot(3, 3, 6)
    success = [1 if r >= SOLVED_SCORE else 0 for r in rewards]
    if len(success) >= window:
        sr = np.convolve(success, np.ones(window)/window, mode='valid') * 100
        plt.plot(sr, color='brown', linewidth=2)
    plt.title('Success Rate (%)', fontweight='bold')
    plt.grid(alpha=0.3)
    
    # 7. Box Plot
    plt.subplot(3, 3, 7)
    n_phases = 5
    segment_size = len(rewards) // n_phases
    segments = [rewards[i*segment_size:(i+1)*segment_size] for i in range(n_phases)]
    plt.boxplot(segments, labels=[f'Phase {i+1}' for i in range(n_phases)])
    plt.title('Performance Evolution', fontweight='bold')
    plt.grid(alpha=0.3)

    # 8. Cumulative
    plt.subplot(3, 3, 8)
    plt.plot(np.cumsum(rewards), color='black', linewidth=2)
    plt.title('Cumulative Reward', fontweight='bold')
    plt.grid(alpha=0.3)

    # 9. Stats
    plt.subplot(3, 3, 9)
    plt.axis('off')
    txt = f"""
    NUCLEAR DQN MOUNTAINCAR
    =======================
    Episodes: {len(rewards)}
    Best 50-Avg: {np.max(np.convolve(rewards, np.ones(50)/50, mode='valid')):.1f}
    Final Epsilon: {epsilons[-1]:.4f}
    Solved Threshold: {SOLVED_SCORE}
    """
    plt.text(0.1, 0.3, txt, fontsize=12, family='monospace')

    plt.tight_layout()
    plt.savefig('dqn_mountaincar_nuclear_dashboard.png', dpi=200)
    print("Graph saved as 'dqn_mountaincar_nuclear_dashboard.png'")
    plt.show()

if __name__ == "__main__":
    r, l, e = train_nuclear_dashboard()
    plot_dashboard(r, l, e)