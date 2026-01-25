import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 1. THE SHARED BRAIN
# ==============================================================================
class ActorCritic(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(ActorCritic, self).__init__()
        # Shared Layer (The Eyes)
        self.affine1 = nn.Linear(n_observations, 128)
        
        # Actor Head (The Chef) - Outputs Probabilities
        self.action_head = nn.Linear(128, n_actions)
        
        # Critic Head (The Judge) - Outputs Value Estimate
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        
        # Actor: Softmax converts numbers to probabilities (0.0 to 1.0)
        action_prob = F.softmax(self.action_head(x), dim=-1)
        
        # Critic: Just a raw number
        state_values = self.value_head(x)
        
        return action_prob, state_values

# ==============================================================================
# 2. HYPERPARAMETERS
# ==============================================================================
LEARNING_RATE = 3e-2
GAMMA = 0.99
EPISODES = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 3. TRAINING LOOP WITH LOGGING
# ==============================================================================
def train_ac_dashboard():
    env = gym.make("CartPole-v1")
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    episode_rewards = []
    losses = []
    actor_losses = []
    critic_losses = []
    
    print(f"Training Actor-Critic on CartPole ({device})...")

    for i_episode in range(EPISODES):
        state, _ = env.reset()
        state = torch.from_numpy(state).float().to(device)
        
        log_probs = []
        values = []
        rewards = []
        
        for t in range(500):
            probs, value = model(state)
            
            # Sample action from probability distribution
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            
            next_state, reward, done, truncated, _ = env.step(action.item())
            
            log_probs.append(m.log_prob(action))
            values.append(value)
            rewards.append(reward)
            
            state = torch.from_numpy(next_state).float().to(device)
            
            if done or truncated:
                break
        
        episode_rewards.append(sum(rewards))

        # --- UPDATE LOGIC ---
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        policy_loss_list = []
        value_loss_list = []
        
        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R - value.item()
            policy_loss_list.append(-log_prob * advantage)
            value_loss_list.append(F.smooth_l1_loss(value, torch.tensor([R]).to(device)))
            
        optimizer.zero_grad()
        
        p_loss = torch.stack(policy_loss_list).sum()
        v_loss = torch.stack(value_loss_list).sum()
        loss = p_loss + v_loss
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        actor_losses.append(p_loss.item())
        critic_losses.append(v_loss.item())

        if i_episode % 50 == 0:
            print(f"Ep {i_episode:3d} | Reward: {sum(rewards):3.0f} | Loss: {loss.item():.2f}")

        # Check for consistent solving (average > 495 over last 100 eps)
        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 495:
             print(f"Mastered at episode {i_episode}! (Avg: {np.mean(episode_rewards[-100:]):.1f})")
             break

    return model, episode_rewards, losses, actor_losses, critic_losses

# ==============================================================================
# 4. DASHBOARD PLOTTING FUNCTION
# ==============================================================================
def plot_dashboard(rewards, losses, a_losses, c_losses):
    print("Generating comprehensive analysis...")
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Raw Rewards
    plt.subplot(3, 3, 1)
    plt.plot(rewards, alpha=0.6, color='blue', linewidth=0.5)
    plt.title('Raw Rewards', fontweight='bold')
    plt.axhline(y=195, color='green', linestyle='--', label='Solved')
    plt.legend()
    plt.grid(alpha=0.3)

    # 2. Moving Average
    plt.subplot(3, 3, 2)
    window = 50
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(ma, color='green', linewidth=2)
    plt.title(f'Moving Average (window={window})', fontweight='bold')
    plt.grid(alpha=0.3)

    # 3. Total Loss
    plt.subplot(3, 3, 3)
    plt.plot(losses, color='purple', alpha=0.6)
    plt.title('Total Loss', fontweight='bold')
    plt.yscale('log')
    plt.grid(alpha=0.3)

    # 4. Actor Loss
    plt.subplot(3, 3, 4)
    plt.plot(a_losses, color='orange', alpha=0.6)
    plt.title('Actor Loss (Policy)', fontweight='bold')
    plt.grid(alpha=0.3)

    # 5. Critic Loss
    plt.subplot(3, 3, 5)
    plt.plot(c_losses, color='teal', alpha=0.6)
    plt.title('Critic Loss (Value)', fontweight='bold')
    plt.grid(alpha=0.3)

    # 6. Reward Distribution
    plt.subplot(3, 3, 6)
    plt.hist(rewards, bins=30, color='skyblue', edgecolor='black')
    plt.title('Reward Distribution', fontweight='bold')
    plt.grid(alpha=0.3)

    # 7. Success Rate
    plt.subplot(3, 3, 7)
    success = [1 if r >= 195 else 0 for r in rewards]
    if len(success) >= window:
        sr = np.convolve(success, np.ones(window)/window, mode='valid') * 100
        plt.plot(sr, color='brown', linewidth=2)
    plt.title('Success Rate (%)', fontweight='bold')
    plt.grid(alpha=0.3)
    
    # 8. Cumulative Reward
    plt.subplot(3, 3, 8)
    plt.plot(np.cumsum(rewards), color='black', linewidth=2)
    plt.title('Cumulative Reward', fontweight='bold')
    plt.grid(alpha=0.3)

    # 9. Stats Text
    plt.subplot(3, 3, 9)
    plt.axis('off')
    txt = f"""
    ACTOR-CRITIC RESULTS
    ====================
    Total Episodes: {len(rewards)}
    Best 100-Avg: {np.max(np.convolve(rewards, np.ones(100)/100, mode='valid')):.1f}
    Final Avg: {np.mean(rewards[-50:]):.1f}
    
    Config:
    - LR: {LEARNING_RATE}
    - Gamma: {GAMMA}
    """
    plt.text(0.1, 0.3, txt, fontsize=12, family='monospace')

    plt.tight_layout()
    plt.savefig('ac_cartpole_dashboard.png', dpi=200)
    print("Graph saved as 'ac_cartpole_dashboard.png'")
    plt.show()

if __name__ == "__main__":
    # Train
    model, r, l, al, cl = train_ac_dashboard()
    
    # Save Model
    torch.save(model.state_dict(), "ac_cartpole.pth")
    print("Model saved as 'ac_cartpole.pth'")
    
    # Plot Dashboard
    plot_dashboard(r, l, al, cl)