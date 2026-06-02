import csv
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

PLT_STYLE = {
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.titlesize': 14,
}
plt.rcParams.update(PLT_STYLE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LR         = 3e-4
GAMMA      = 0.99
N_EPISODES = 2000
MAX_STEPS  = 500
SOLVED_AVG = 195    # standard CartPole solved criterion


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
    env       = gym.make('CartPole-v1')
    env.reset(seed=SEED)
    model     = ActorCritic(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    episode_rewards = []
    actor_losses    = []
    critic_losses   = []
    solve_ep        = None

    print(f"CartPole Actor-Critic | Device: {DEVICE}")
    print("=" * 50)

    for episode in range(N_EPISODES):
        state, _ = env.reset()
        state    = torch.from_numpy(state).float().to(DEVICE)
        log_probs, values, rewards = [], [], []

        for _ in range(MAX_STEPS):
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

        returns, R = [], 0.0
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        values_t  = torch.cat(values).squeeze(-1)

        advantages = returns_t - values_t.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
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
            print(f"Episode {episode+1:4d} | Avg(100): {avg:6.2f} | Loss: {loss.item():.3f}")

        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= SOLVED_AVG:
            if solve_ep is None:
                solve_ep = episode + 1
            print(f"\nSolved at episode {solve_ep}! Avg(100): {np.mean(episode_rewards[-100:]):.2f}")
            break

    env.close()
    return model, episode_rewards, actor_losses, critic_losses, solve_ep


def save_log(episode_rewards, actor_losses, critic_losses, save_dir):
    path = f'{save_dir}/training_log.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'actor_loss', 'critic_loss', 'total_loss'])
        for i in range(len(episode_rewards)):
            total = actor_losses[i] + critic_losses[i]
            writer.writerow([i + 1, episode_rewards[i],
                             round(actor_losses[i], 6), round(critic_losses[i], 6),
                             round(total, 6)])
    print(f"Saved: {path}")


def save_summary(episode_rewards, actor_losses, critic_losses, solve_ep, save_dir):
    n        = len(episode_rewards)
    last_100 = episode_rewards[-100:] if n >= 100 else episode_rewards
    success  = sum(1 for r in last_100 if r >= 195)

    lines = [
        "=" * 55,
        "TRAINING SUMMARY — CartPole Actor-Critic (A2C)",
        "=" * 55,
        f"  Episodes trained         : {n:,}",
        f"  Solved at episode        : {solve_ep if solve_ep else 'Not solved'}",
        f"  Mean reward (all)        : {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}",
        f"  Mean reward (last 100)   : {np.mean(last_100):.2f} ± {np.std(last_100):.2f}",
        f"  Best single episode      : {int(np.max(episode_rewards))}",
        f"  Success rate ≥195 (last100): {success} / {len(last_100)}  ({100*success/len(last_100):.1f}%)",
        f"  Final mean actor loss    : {np.mean(actor_losses[-50:]):.4f}",
        f"  Final mean critic loss   : {np.mean(critic_losses[-50:]):.4f}",
        "",
        "  Hyperparameters",
        f"    Learning rate          : {LR}",
        f"    Gamma                  : {GAMMA}",
        f"    Gradient clip norm     : 0.5",
        f"    Network                : 4 → 128 (shared) → actor(2) + critic(1)",
        f"    Optimiser              : Adam",
        f"    Return normalisation   : per-episode z-score",
        "=" * 55,
    ]
    text = '\n'.join(lines)
    print('\n' + text)
    with open(f'{save_dir}/summary.txt', 'w') as f:
        f.write(text + '\n')
    print(f"Saved: {save_dir}/summary.txt")


def plot_results(episode_rewards, actor_losses, critic_losses, save_dir):
    window = 50
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Raw rewards
    axes[0, 0].plot(episode_rewards, alpha=0.4, color='steelblue', linewidth=0.6, label='Per episode')
    axes[0, 0].axhline(y=195, color='red', linestyle='--', linewidth=1.5, label='Basic solve (195)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward (= steps)')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Moving average
    if len(episode_rewards) >= window:
        ma = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        axes[0, 1].plot(range(window - 1, len(episode_rewards)), ma, color='green', linewidth=2)
        axes[0, 1].axhline(y=195, color='red', linestyle='--', linewidth=1.5, label='Basic solve (195)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].set_title(f'Moving Average ({window} episodes)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Actor loss
    axes[0, 2].plot(actor_losses, alpha=0.5, color='darkorange', linewidth=0.8, label='Actor loss')
    if len(actor_losses) >= window:
        ma_a = np.convolve(actor_losses, np.ones(window) / window, mode='valid')
        axes[0, 2].plot(range(window - 1, len(actor_losses)), ma_a, color='saddlebrown', linewidth=1.5, label=f'{window}-ep avg')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('Actor (Policy) Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Critic loss
    axes[1, 0].plot(critic_losses, alpha=0.5, color='teal', linewidth=0.8, label='Critic loss')
    if len(critic_losses) >= window:
        ma_c = np.convolve(critic_losses, np.ones(window) / window, mode='valid')
        axes[1, 0].plot(range(window - 1, len(critic_losses)), ma_c, color='darkcyan', linewidth=1.5, label=f'{window}-ep avg')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Critic (Value) Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Reward distribution
    axes[1, 1].hist(episode_rewards, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    axes[1, 1].axvline(x=195, color='red', linestyle='--', linewidth=1.5, label='Solve (195)')
    axes[1, 1].axvline(x=np.mean(episode_rewards), color='orange', linewidth=1.5,
                        label=f'Mean: {np.mean(episode_rewards):.1f}')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # 6. Summary text
    n        = len(episode_rewards)
    last_100 = episode_rewards[-100:] if n >= 100 else episode_rewards
    axes[1, 2].axis('off')
    stats = (
        f"Episodes         : {n}\n"
        f"Last-100 mean    : {np.mean(last_100):.1f}\n"
        f"Last-100 std     : {np.std(last_100):.1f}\n"
        f"Best episode     : {int(np.max(episode_rewards))}\n\n"
        f"Config\n"
        f"  LR             : {LR}\n"
        f"  Gamma          : {GAMMA}\n"
        f"  Grad clip      : 0.5\n"
        f"  Net            : 4→128→actor+critic"
    )
    axes[1, 2].text(0.05, 0.95, stats, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    axes[1, 2].set_title('Run Statistics')

    plt.suptitle('CartPole Actor-Critic — Training Analysis', fontsize=14, fontweight='bold')
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

        for _ in range(MAX_STEPS):
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
    print(f"\n  Average : {avg:.2f} ± {np.std(test_rewards):.2f}")
    print(f"  Result  : {'SOLVED' if avg >= 195 else 'Needs more training'}")
    env.close()


if __name__ == "__main__":
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/cartpole_ac_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    model, rewards, a_losses, c_losses, solve_ep = train()

    save_log(rewards, a_losses, c_losses, results_dir)
    save_summary(rewards, a_losses, c_losses, solve_ep, results_dir)
    plot_results(rewards, a_losses, c_losses, results_dir)
    test_agent(model)

    torch.save(model.state_dict(), f'{results_dir}/model.pth')
    print(f"\nResults saved to: {results_dir}/")
