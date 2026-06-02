import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import os
import gymnasium as gym

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LR         = 3e-4   # lowered from 3e-2 for stability
GAMMA      = 0.99
N_EPISODES = 1000
SOLVED_AVG = 495    # CartPole "mastered" threshold (max episode length)


class ActorCritic(nn.Module):
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.shared = nn.Linear(n_obs, 128)
        self.actor  = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.shared(x))
        return F.softmax(self.actor(x), dim=-1), self.critic(x)


def train():
    env   = gym.make('CartPole-v1')
    env.reset(seed=SEED)
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    episode_rewards = []
    actor_losses    = []
    critic_losses   = []

    print(f"CartPole Actor-Critic | Device: {DEVICE}")
    print("=" * 50)

    for episode in range(N_EPISODES):
        state, _ = env.reset()
        state = torch.from_numpy(state).float().to(DEVICE)

        log_probs, values, rewards = [], [], []

        for _ in range(500):
            probs, value = model(state)
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)

            state = torch.from_numpy(next_obs).float().to(DEVICE)
            if terminated or truncated:
                break

        episode_rewards.append(sum(rewards))

        # Compute discounted returns
        returns, R = [], 0.0
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-9)
        values_t  = torch.cat(values).squeeze()

        advantages  = returns_t - values_t.detach()
        policy_loss = -(torch.stack(log_probs) * advantages).sum()
        value_loss  = F.smooth_l1_loss(values_t, returns_t)
        loss        = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        actor_losses.append(policy_loss.item())
        critic_losses.append(value_loss.item())

        if (episode + 1) % 50 == 0:
            avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"Episode {episode+1:4d} | Avg(100): {avg:6.2f} | Loss: {loss.item():.2f}")

        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= SOLVED_AVG:
            print(f"\nSolved at episode {episode+1}! Avg(100): {np.mean(episode_rewards[-100:]):.2f}")
            break

    env.close()
    return model, episode_rewards, actor_losses, critic_losses


def plot_results(episode_rewards, actor_losses, critic_losses, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(episode_rewards, alpha=0.5, color='blue', linewidth=0.5)
    axes[0, 0].axhline(y=195, color='red', linestyle='--', alpha=0.5, label='Basic solve (195)')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    window = 50
    if len(episode_rewards) >= window:
        ma = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        axes[0, 1].plot(range(window - 1, len(episode_rewards)), ma, color='green', linewidth=2)
        axes[0, 1].axhline(y=195, color='red', linestyle='--', alpha=0.5, label='Basic solve (195)')
    axes[0, 1].set_title(f'Moving Average ({window} episodes)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(actor_losses, color='orange', alpha=0.7, linewidth=0.8)
    axes[1, 0].set_title('Actor (Policy) Loss')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(critic_losses, color='teal', alpha=0.7, linewidth=0.8)
    axes[1, 1].set_title('Critic (Value) Loss')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('CartPole Actor-Critic', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/training_results.png")


def test_agent(model, n_episodes=10):
    env = gym.make('CartPole-v1')
    test_rewards = []
    model.eval()

    print(f"\nTesting trained agent ({n_episodes} episodes)...")
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(500):
            state_t = torch.from_numpy(state).float().to(DEVICE)
            with torch.no_grad():
                probs, _ = model(state_t)
            action = probs.argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        test_rewards.append(total_reward)
        status = "✓" if total_reward >= 195 else "✗"
        print(f"  {status} Test {episode+1:2d}: {total_reward:.0f} steps")

    avg = np.mean(test_rewards)
    print(f"\n  Average: {avg:.2f}")
    print(f"  {'Solved!' if avg >= 195 else 'Needs more training.'}")
    env.close()


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/cartpole_ac_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    model, rewards, a_losses, c_losses = train()
    plot_results(rewards, a_losses, c_losses, results_dir)
    test_agent(model)

    torch.save(model.state_dict(), f'{results_dir}/model.pth')
    print(f"\nResults saved to: {results_dir}/")
