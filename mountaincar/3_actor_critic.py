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
N_EPISODES = 3000
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
        return torch.softmax(self.actor(x), dim=-1), self.critic(x)


def normalize_state(state):
    s    = np.array(state, dtype=np.float32)
    s[0] = (s[0] + 0.3) / 0.9   # maps [-1.2, 0.6] → [-1, 1]
    s[1] = s[1] / 0.07            # maps [-0.07, 0.07] → [-1, 1]
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
    env       = gym.make('MountainCar-v0')
    env.reset(seed=SEED)
    model     = ActorCritic(2, env.action_space.n).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    episode_raw_rewards = []
    episode_lengths     = []
    actor_losses        = []
    critic_losses       = []
    solve_ep            = None

    print(f"MountainCar Actor-Critic | Device: {DEVICE}")
    print("=" * 55)

    for episode in range(N_EPISODES):
        state, _  = env.reset()
        log_probs, values, s_rewards = [], [], []
        raw_total = 0
        steps     = 0

        for _ in range(MAX_STEPS):
            state_norm = normalize_state(state)
            state_t    = torch.from_numpy(state_norm).float().unsqueeze(0).to(DEVICE)

            probs, value = model(state_t)
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, raw_reward, terminated, truncated, _ = env.step(action.item())
            done  = terminated or truncated
            s_rew = shaped_reward(next_state[0], next_state[1], terminated)

            log_probs.append(dist.log_prob(action))
            values.append(value)
            s_rewards.append(s_rew)

            state      = next_state
            raw_total += raw_reward
            steps     += 1
            if done:
                break

        episode_raw_rewards.append(raw_total)
        episode_lengths.append(steps)

        returns, R = [], 0.0
        for r in reversed(s_rewards):
            R = r + GAMMA * R
            returns.insert(0, R)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-9)
        values_t  = torch.cat(values).squeeze(-1)

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
            print(f"Episode {episode+1:4d} | Avg raw(100): {avg_raw:7.2f} | "
                  f"Success: {success:3d}/100 | Steps: {steps:3d}")

        if steps < MAX_STEPS:
            print(f"  >>> Episode {episode+1} REACHED GOAL in {steps} steps!")

        if len(episode_raw_rewards) >= 100 and np.mean(episode_raw_rewards[-100:]) >= SOLVED_AVG:
            if solve_ep is None:
                solve_ep = episode + 1
            print(f"\nSolved at episode {solve_ep}! Avg raw(100): {np.mean(episode_raw_rewards[-100:]):.2f}")
            break

    env.close()
    return model, episode_raw_rewards, episode_lengths, actor_losses, critic_losses, solve_ep


def save_log(episode_raw_rewards, episode_lengths, actor_losses, critic_losses, save_dir):
    path = f'{save_dir}/training_log.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'raw_reward', 'steps', 'actor_loss', 'critic_loss'])
        for i in range(len(episode_raw_rewards)):
            writer.writerow([i + 1, episode_raw_rewards[i], episode_lengths[i],
                             round(actor_losses[i], 6), round(critic_losses[i], 6)])
    print(f"Saved: {path}")


def save_summary(episode_raw_rewards, episode_lengths, actor_losses, critic_losses, solve_ep, save_dir):
    n        = len(episode_raw_rewards)
    last_100 = episode_raw_rewards[-100:] if n >= 100 else episode_raw_rewards
    success  = sum(1 for r in last_100 if r >= SOLVED_AVG)

    lines = [
        "=" * 55,
        "TRAINING SUMMARY — MountainCar Actor-Critic (A2C)",
        "=" * 55,
        f"  Episodes trained         : {n:,}",
        f"  Solved at episode        : {solve_ep if solve_ep else 'Not solved'}",
        f"  Mean reward (all)        : {np.mean(episode_raw_rewards):.2f} ± {np.std(episode_raw_rewards):.2f}",
        f"  Mean reward (last 100)   : {np.mean(last_100):.2f} ± {np.std(last_100):.2f}",
        f"  Success rate (last 100)  : {success} / {len(last_100)}  ({100*success/len(last_100):.1f}%)",
        f"  Mean steps (last 100)    : {np.mean(episode_lengths[-100:]):.1f}",
        f"  Final mean actor loss    : {np.mean(actor_losses[-50:]):.4f}",
        f"  Final mean critic loss   : {np.mean(critic_losses[-50:]):.4f}",
        "",
        "  Hyperparameters",
        f"    Learning rate          : {LR}",
        f"    Gamma                  : {GAMMA}",
        f"    Gradient clip norm     : 0.5",
        f"    Network hidden         : 64 units, 2 layers (Tanh)",
        f"    Initialisation         : orthogonal",
        f"    State normalisation    : pos → [-1,1], vel → [-1,1]",
        f"    Reward shaping         : height + 100·KE - 1  (+10 at goal)",
        "=" * 55,
    ]
    text = '\n'.join(lines)
    print('\n' + text)
    with open(f'{save_dir}/summary.txt', 'w') as f:
        f.write(text + '\n')
    print(f"Saved: {save_dir}/summary.txt")


def plot_results(episode_raw_rewards, episode_lengths, actor_losses, critic_losses, save_dir):
    window = 100
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Raw rewards
    axes[0, 0].plot(episode_raw_rewards, alpha=0.4, color='steelblue', linewidth=0.5, label='Per episode')
    axes[0, 0].axhline(y=SOLVED_AVG, color='red', linestyle='--', linewidth=1.5, label=f'Solved ({SOLVED_AVG})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward (real score)')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Moving average
    if len(episode_raw_rewards) >= window:
        ma = np.convolve(episode_raw_rewards, np.ones(window) / window, mode='valid')
        axes[0, 1].plot(range(window - 1, len(episode_raw_rewards)), ma, color='green', linewidth=2, label=f'{window}-ep avg')
        axes[0, 1].axhline(y=SOLVED_AVG, color='red', linestyle='--', linewidth=1.5)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].set_title(f'Moving Average ({window} episodes)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Steps per episode
    axes[0, 2].plot(episode_lengths, alpha=0.4, color='darkorange', linewidth=0.5)
    axes[0, 2].axhline(y=110, color='green', linestyle='--', linewidth=1.5, label='Target (≤110)')
    if len(episode_lengths) >= window:
        ma_len = np.convolve(episode_lengths, np.ones(window) / window, mode='valid')
        axes[0, 2].plot(range(window - 1, len(episode_lengths)), ma_len, color='saddlebrown', linewidth=2)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Steps')
    axes[0, 2].set_title('Steps per Episode')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Actor loss
    w = 50
    axes[1, 0].plot(actor_losses, alpha=0.4, color='darkorange', linewidth=0.6)
    if len(actor_losses) >= w:
        ma_a = np.convolve(actor_losses, np.ones(w) / w, mode='valid')
        axes[1, 0].plot(range(w - 1, len(actor_losses)), ma_a, color='saddlebrown', linewidth=1.5, label=f'{w}-ep avg')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Actor (Policy) Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Critic loss
    axes[1, 1].plot(critic_losses, alpha=0.4, color='teal', linewidth=0.6)
    if len(critic_losses) >= w:
        ma_c = np.convolve(critic_losses, np.ones(w) / w, mode='valid')
        axes[1, 1].plot(range(w - 1, len(critic_losses)), ma_c, color='darkcyan', linewidth=1.5, label=f'{w}-ep avg')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Critic (Value) Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Reward distribution
    axes[1, 2].hist(episode_raw_rewards, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    axes[1, 2].axvline(x=SOLVED_AVG, color='red', linestyle='--', linewidth=1.5, label=f'Solved ({SOLVED_AVG})')
    axes[1, 2].axvline(x=np.mean(episode_raw_rewards), color='orange', linewidth=1.5,
                        label=f'Mean: {np.mean(episode_raw_rewards):.1f}')
    axes[1, 2].set_xlabel('Reward')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Reward Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('MountainCar Actor-Critic — Training Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/training_results.png")


def plot_trajectory(model, save_dir):
    """Run one greedy episode and plot position + velocity over time."""
    env = gym.make('MountainCar-v0')
    state, _ = env.reset(seed=0)
    positions, velocities = [], []
    model.eval()

    for _ in range(MAX_STEPS):
        positions.append(state[0])
        velocities.append(state[1])
        state_norm = normalize_state(state)
        state_t    = torch.from_numpy(state_norm).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs, _ = model(state_t)
        action = probs.argmax().item()
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    positions.append(state[0])
    velocities.append(state[1])
    env.close()

    reached   = max(positions) >= 0.5
    status    = f"REACHED GOAL in {len(positions)-1} steps" if reached else f"DID NOT REACH GOAL ({len(positions)-1} steps)"
    timesteps = range(len(positions))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(timesteps, positions, color='steelblue', linewidth=1.8)
    axes[0].axhline(y=0.5,  color='green', linestyle='--', linewidth=1.5, label='Goal (pos = 0.5)')
    axes[0].axhline(y=-0.5, color='gray',  linestyle=':', alpha=0.5,      label='Valley bottom (-0.5)')
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('Position')
    axes[0].set_title('Car Position Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(timesteps, velocities, color='darkorange', linewidth=1.8)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Zero velocity')
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Velocity')
    axes[1].set_title('Car Velocity Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Test Episode Trajectory — {status}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/trajectory.png")


def test_agent(model, n_episodes=20):
    env = gym.make('MountainCar-v0')
    test_rewards, test_lengths = [], []
    model.eval()

    print(f"\nTesting trained agent ({n_episodes} episodes)...")
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps        = 0

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
        test_lengths.append(steps)
        status = "✓" if total_reward >= SOLVED_AVG else "✗"
        print(f"  {status} Test {episode+1:2d}: {total_reward:6.1f}  ({steps:3d} steps)")

    avg     = np.mean(test_rewards)
    success = sum(1 for r in test_rewards if r >= SOLVED_AVG)
    print(f"\n  Average : {avg:.2f} ± {np.std(test_rewards):.2f}")
    print(f"  Success : {success}/{n_episodes}  ({100*success/n_episodes:.0f}%)")
    print(f"  Result  : {'SOLVED' if avg >= SOLVED_AVG else 'Needs more training'}")
    env.close()


if __name__ == "__main__":
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/mountaincar_ac_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    model, raw_rewards, lengths, a_losses, c_losses, solve_ep = train()

    save_log(raw_rewards, lengths, a_losses, c_losses, results_dir)
    save_summary(raw_rewards, lengths, a_losses, c_losses, solve_ep, results_dir)
    plot_results(raw_rewards, lengths, a_losses, c_losses, results_dir)
    plot_trajectory(model, results_dir)
    test_agent(model)

    torch.save(model.state_dict(), f'{results_dir}/model.pth')
    print(f"\nResults saved to: {results_dir}/")
