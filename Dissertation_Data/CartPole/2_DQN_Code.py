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
# 1. THE NEURAL NETWORK (Your exact structure)
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
# 2. HYPERPARAMETERS (Your exact values)
# ==============================================================================
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 3. TRAINING FUNCTION (Added Data Logging Only)
# ==============================================================================
def train_dqn_with_logging():
    env = gym.make("CartPole-v1")
    n_actions = env.action_space.n
    n_observations = env.observation_space.shape[0]

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = deque(maxlen=10000)

    steps_done = 0
    
    # --- NEW: Lists to store data for graphs ---
    episode_rewards = []
    loss_history = []
    epsilon_history = []

    print("Training Started... (Using your verified logic)")
    
    for i_episode in range(600):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Trackers for this specific episode
        current_episode_loss = []
        
        for t in range(500):
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
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.append((state, action, next_state, reward))
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
                
                # --- NEW: Record the loss ---
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
        
        # --- NEW: Save Episode Stats ---
        episode_rewards.append(t + 1)
        epsilon_history.append(eps_threshold)
        if current_episode_loss:
            loss_history.append(np.mean(current_episode_loss))
        else:
            loss_history.append(0)

        if i_episode % 50 == 0:
            print(f"Episode {i_episode} finished. Duration: {t+1}")

    print("Training Complete!")
    return policy_net, episode_rewards, loss_history, epsilon_history

# ==============================================================================
# 4. PLOTTING FUNCTION (The Dashboard)
# ==============================================================================
def plot_dashboard(rewards, losses, epsilons):
    print("Generating comprehensive analysis...")
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Raw Rewards
    plt.subplot(3, 3, 1)
    plt.plot(rewards, alpha=0.6, color='blue', linewidth=0.5)
    plt.title('Raw Rewards per Episode', fontweight='bold')
    plt.ylabel('Reward')
    plt.axhline(y=195, color='red', linestyle='--', label='Solved')
    plt.legend()
    plt.grid(alpha=0.3)

    # 2. Moving Average
    plt.subplot(3, 3, 2)
    window = 50
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(ma, color='green', linewidth=2)
    plt.title(f'Moving Average (window={window})', fontweight='bold')
    plt.axhline(y=195, color='red', linestyle='--', label='Solved')
    plt.grid(alpha=0.3)

    # 3. Epsilon Decay
    plt.subplot(3, 3, 3)
    plt.plot(epsilons, color='orange', linewidth=2)
    plt.title('Exploration Rate', fontweight='bold')
    plt.ylabel('Epsilon')
    plt.grid(alpha=0.3)

    # 4. LOSS CURVE
    plt.subplot(3, 3, 4)
    plt.plot(losses, color='purple', alpha=0.7, linewidth=1)
    if len(losses) > 50:
        ma_loss = np.convolve(losses, np.ones(50)/50, mode='valid')
        plt.plot(ma_loss, color='black', linewidth=1.5, label='Trend')
    plt.title('Training Loss', fontweight='bold')
    plt.ylabel('Loss (Huber)')
    plt.yscale('log')
    plt.legend()
    plt.grid(alpha=0.3)

    # 5. Reward Dist
    plt.subplot(3, 3, 5)
    plt.hist(rewards, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=195, color='red', linestyle='--', label='Solved')
    plt.title('Distribution of Rewards', fontweight='bold')
    plt.grid(alpha=0.3)

    # 6. Box Plot Phases
    plt.subplot(3, 3, 6)
    n_phases = 5
    segment_size = len(rewards) // n_phases
    segments = [rewards[i*segment_size:(i+1)*segment_size] for i in range(n_phases)]
    plt.boxplot(segments, labels=[f'Phase {i+1}' for i in range(n_phases)])
    plt.title('Performance Phases', fontweight='bold')
    plt.grid(alpha=0.3)

    # 7. Success Rate
    plt.subplot(3, 3, 7)
    success = [1 if r >= 195 else 0 for r in rewards]
    if len(success) >= window:
        success_rate = np.convolve(success, np.ones(window)/window, mode='valid') * 100
        plt.plot(success_rate, color='teal', linewidth=2)
    plt.title('Success Rate (%)', fontweight='bold')
    plt.grid(alpha=0.3)

    # 8. Cumulative
    plt.subplot(3, 3, 8)
    plt.plot(np.cumsum(rewards), color='brown', linewidth=2)
    plt.title('Cumulative Reward', fontweight='bold')
    plt.grid(alpha=0.3)

    # 9. Stats
    plt.subplot(3, 3, 9)
    plt.axis('off')
    txt = f"""
    FINAL STATISTICS
    ================
    Episodes: {len(rewards)}
    Best 100-avg: {np.max(np.convolve(rewards, np.ones(100)/100, mode='valid')):.1f}
    Final Epsilon: {epsilons[-1]:.4f}
    
    Hyperparameters:
    - Batch: {BATCH_SIZE}
    - Gamma: {GAMMA}
    - LR: {LR}
    - Memory: 10000
    """
    plt.text(0.1, 0.3, txt, fontsize=12, family='monospace')

    plt.tight_layout()
    plt.savefig('dqn_final_analysis.png', dpi=200)
    print("Graph saved as 'dqn_final_analysis.png'")
    plt.show()

if __name__ == "__main__":
    # 1. Train with your trusted logic
    model, rewards, losses, epsilons = train_dqn_with_logging()
    
    # 2. Save
    torch.save(model.state_dict(), "dqn_cartpole_final.pth")
    print("Model saved.")
    
    # 3. Plot
    plot_dashboard(rewards, losses, epsilons)