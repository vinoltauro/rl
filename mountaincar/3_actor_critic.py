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
LR         = 3e-4
GAMMA      = 0.99
N_EPISODES = 2000
MAX_STEPS  = 200
SOLVED_AVG = -110.0


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),     nn.Tanh(),
            nn.Linear(hidden, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),     nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        logits = self.actor(x)
        probs  = torch.softmax(logits, dim=-1)
        value  = self.critic(x)
        return probs, value


def normalize_state(state):
    s = np.array(state, dtype=np.float32)
    s[0] = (s[0] + 0.3) / 0.9   # centre and scale position
    s[1] = s[1] / 0.07           # scale velocity
    return s


def shaped_reward(position, velocity, terminated):
    """Height + kinetic energy bonus to overcome sparse reward."""
    height = (position + 1.2) / 1.8 * 2.0
    ke     = velocity * velocity
    reward = height + 100.0 * ke - 1.0
    if terminated and position >= 0.5:
        reward += 10.0
    return reward


def train():
    env   = gym.make('MountainCar-v0')
    env.reset(seed=SEED)
    model = ActorCritic(2, env.action_space.n).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    episode_rewards     = []
    episode_raw_rewards = []
    actor_losses        = []
    critic_losses       = []

    print(f"MountainCar Actor-Critic | Device: {DEVICE}")
    print("=" * 50)

    for episode in range(N_EPISODES):
        state, _ = env.reset()
        log_probs, values, s_rewards = [], [], []
        raw_reward_total = 0
        steps = 0

        for _ in range(MAX_STEPS):
            state_norm = normalize_state(state)
            state_t    = torch.from_numpy(state_norm).float().unsqueeze(0).to(DEVICE)

            probs, value = model(state_t)
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, raw_reward, terminated, truncated, _ = env.step(action.item())
            done          = terminated or truncated
            s_rew         = shaped_reward(next_state[0], next_state[1], terminated)

            log_probs.append(dist.log_prob(action))
            values.append(value)
            s_rewards.append(s_rew)

            state             = next_state
            raw_reward_total += raw_reward
            steps            += 1
            if done:
                break

        episode_rewards.append(sum(s_rewards))
        episode_raw_rewards.append(raw_reward_total)

        # Compute discounted returns
        returns, R = [], 0.0
        for r in reversed(s_rewards):
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

        if (episode + 1) % 100 == 0:
            avg_raw = np.mean(episode_raw_rewards[-100:]) if len(episode_raw_rewards) >= 100 else np.mean(episode_raw_rewards)
            success = sum(1 for r in episode_raw_rewards[-100:] if r >= SOLVED_AVG)
            print(f"Episode {episode+1:4d} | Avg raw (100): {avg_raw:7.2f} | "
                  f"Success: {success:3d}/100 | Steps: {steps:3d}")

        if steps < MAX_STEPS:
            print(f"  >>> Episode {episode+1} REACHED GOAL in {steps} steps!")

        if len(episode_raw_rewards) >= 100 and np.mean(episode_raw_rewards[-100:]) >= SOLVED_AVG:
            print(f"\nSolved at episode {episode+1}! Avg raw(100): {np.mean(episode_raw_rewards[-100:]):.2f}")
            break

    env.close()
    return model, episode_raw_rewards, actor_losses, critic_losses


def plot_results(episode_raw_rewards, actor_losses, critic_losses, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    window = 100

    axes[0, 0].plot(episode_raw_rewards, alpha=0.5, color='blue', linewidth=0.5)
    axes[0, 0].axhline(y=SOLVED_AVG, color='red', linestyle='--', label=f'Solved ({SOLVED_AVG})')
    axes[0, 0].set_title('Episode Rewards (real score)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    if len(episode_raw_rewards) >= window:
        ma = np.convolve(episode_raw_rewards, np.ones(window) / window, mode='valid')
        axes[0, 1].plot(range(window - 1, len(episode_raw_rewards)), ma, color='green', linewidth=2)
        axes[0, 1].axhline(y=SOLVED_AVG, color='red', linestyle='--', label=f'Solved ({SOLVED_AVG})')
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

    plt.suptitle('MountainCar Actor-Critic (reward shaping)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/training_results.png")


def test_agent(model, n_episodes=20):
    env = gym.make('MountainCar-v0')
    test_rewards = []
    model.eval()

    print(f"\nTesting trained agent ({n_episodes} episodes)...")
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        for _ in range(MAX_STEPS):
            state_norm = normalize_state(state)
            state_t    = torch.from_numpy(state_norm).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs, _ = model(state_t)
            action = probs.argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        test_rewards.append(total_reward)
        status = "✓" if total_reward >= SOLVED_AVG else "✗"
        print(f"  {status} Test {episode+1:2d}: {total_reward:6.1f}  ({steps:3d} steps)")

    avg     = np.mean(test_rewards)
    success = sum(1 for r in test_rewards if r >= SOLVED_AVG)
    print(f"\n  Average: {avg:.2f}  |  Success: {success}/{n_episodes}")
    print(f"  {'Solved!' if avg >= SOLVED_AVG else 'Needs more training.'}")
    env.close()


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/mountaincar_ac_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    model, rewards, a_losses, c_losses = train()
    plot_results(rewards, a_losses, c_losses, results_dir)
    test_agent(model)

    torch.save(model.state_dict(), f'{results_dir}/model.pth')
    print(f"\nResults saved to: {results_dir}/")
