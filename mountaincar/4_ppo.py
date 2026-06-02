import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
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
LR           = 3e-4
GAMMA        = 0.99
LAM          = 0.95
CLIP_EPS     = 0.2
ENTROPY_COEF = 0.05
VALUE_COEF   = 0.5
EPOCHS       = 10
BATCH_SIZE   = 64
UPDATE_FREQ  = 1024   # steps between updates — more frequent updates for sparse reward env
N_EPISODES   = 3000
MAX_STEPS    = 200
SOLVED_AVG   = -110.0


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

    def forward(self, state):
        return torch.softmax(self.actor(state), dim=-1), self.critic(state)


class PPOAgent:
    def __init__(self, state_size, action_size):
        self.policy    = ActorCritic(state_size, action_size).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR, eps=1e-5)
        self._clear()

    def _clear(self):
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.values    = []
        self.log_probs = []
        self.dones     = []

    def normalize_state(self, state):
        s    = np.array(state, dtype=np.float32)
        s[0] = (s[0] + 0.3) / 0.9
        s[1] = s[1] / 0.07
        return s

    def select_action(self, state, deterministic=False):
        s = torch.FloatTensor(self.normalize_state(state)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs, value = self.policy(s)
        if deterministic:
            return probs.argmax().item(), 0.0, 0.0
        dist   = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def store(self, state, action, reward, log_prob, value, done):
        self.states.append(self.normalize_state(state))
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def update(self):
        if not self.states:
            return 0, 0, 0

        last_s = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, next_value = self.policy(last_s)
            next_value = next_value.item()

        advantages, gae = [], 0
        values = self.values + [next_value]
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + GAMMA * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae   = delta + GAMMA * LAM * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        states     = torch.FloatTensor(np.array(self.states)).to(DEVICE)
        actions    = torch.LongTensor(self.actions).to(DEVICE)
        old_lp     = torch.FloatTensor(self.log_probs).to(DEVICE)
        advantages = torch.FloatTensor(advantages).to(DEVICE)
        returns    = advantages + torch.FloatTensor(self.values).to(DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(len(states))
        total_pl, total_vl, total_ent, n = 0, 0, 0, 0

        for _ in range(EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, len(states), BATCH_SIZE):
                idx       = indices[start:start + BATCH_SIZE]
                b_states  = states[idx]
                b_actions = actions[idx]
                b_old_lp  = old_lp[idx]
                b_adv     = advantages[idx]
                b_ret     = returns[idx]

                probs, values = self.policy(b_states)
                dist          = Categorical(probs)
                new_lp        = dist.log_prob(b_actions)
                entropy       = dist.entropy().mean()

                ratio = torch.exp(new_lp - b_old_lp)
                pl    = -torch.min(ratio * b_adv,
                                   torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * b_adv).mean()
                vl    = 0.5 * ((values.squeeze() - b_ret) ** 2).mean()
                loss  = pl + VALUE_COEF * vl - ENTROPY_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_pl  += pl.item()
                total_vl  += vl.item()
                total_ent += entropy.item()
                n         += 1

        self._clear()
        return total_pl / n, total_vl / n, total_ent / n


def shaped_reward(position, velocity, terminated):
    height = (position + 1.2) / 1.8 * 2.0
    ke     = velocity * velocity
    reward = height + 100.0 * ke - 1.0
    if terminated and position >= 0.5:
        reward += 10.0
    return reward


def train():
    env   = gym.make('MountainCar-v0')
    env.reset(seed=SEED)
    agent = PPOAgent(2, env.action_space.n)

    episode_raw_rewards = []
    episode_lengths     = []
    policy_losses       = []
    value_losses        = []
    entropies           = []
    solve_ep            = None

    print(f"MountainCar PPO | Device: {DEVICE}")
    print("=" * 55)

    for episode in range(N_EPISODES):
        state, _      = env.reset()
        ep_raw_reward = 0
        steps         = 0

        for _ in range(MAX_STEPS):
            action, log_prob, val = agent.select_action(state)
            next_state, raw_reward, terminated, truncated, _ = env.step(action)
            done  = terminated or truncated
            s_rew = shaped_reward(next_state[0], next_state[1], terminated)

            agent.store(state, action, s_rew, log_prob, val, done)
            state          = next_state
            ep_raw_reward += raw_reward
            steps         += 1

            if len(agent.states) >= UPDATE_FREQ:
                pl, vl, ent = agent.update()
                policy_losses.append(pl)
                value_losses.append(vl)
                entropies.append(ent)

            if done:
                break

        episode_raw_rewards.append(ep_raw_reward)
        episode_lengths.append(steps)

        if (episode + 1) % 50 == 0:
            avg_raw = np.mean(episode_raw_rewards[-100:]) if len(episode_raw_rewards) >= 100 else np.mean(episode_raw_rewards)
            success = sum(1 for r in episode_raw_rewards[-100:] if r >= SOLVED_AVG)
            print(f"Episode {episode+1:4d} | Avg raw(100): {avg_raw:7.2f} | Success: {success:3d}/100 | Steps: {steps:3d}")

        if steps < MAX_STEPS:
            print(f"  >>> Episode {episode+1} REACHED GOAL in {steps} steps!")

        if len(episode_raw_rewards) >= 100 and np.mean(episode_raw_rewards[-100:]) >= SOLVED_AVG:
            if solve_ep is None:
                solve_ep = episode + 1
            print(f"\nSolved at episode {solve_ep}! Avg raw(100): {np.mean(episode_raw_rewards[-100:]):.2f}")
            break

    env.close()
    return agent, episode_raw_rewards, episode_lengths, policy_losses, value_losses, entropies, solve_ep


def save_log(episode_raw_rewards, episode_lengths, policy_losses, value_losses, entropies, save_dir):
    path = f'{save_dir}/training_log.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'raw_reward', 'steps'])
        for i in range(len(episode_raw_rewards)):
            writer.writerow([i + 1, episode_raw_rewards[i], episode_lengths[i]])
    print(f"Saved: {path}")

    if policy_losses:
        path2 = f'{save_dir}/update_log.csv'
        with open(path2, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['update', 'policy_loss', 'value_loss', 'entropy'])
            for i in range(len(policy_losses)):
                writer.writerow([i + 1, round(policy_losses[i], 6),
                                 round(value_losses[i], 6), round(entropies[i], 6)])
        print(f"Saved: {path2}")


def save_summary(episode_raw_rewards, episode_lengths, policy_losses, value_losses,
                 entropies, solve_ep, save_dir):
    n        = len(episode_raw_rewards)
    last_100 = episode_raw_rewards[-100:] if n >= 100 else episode_raw_rewards
    success  = sum(1 for r in last_100 if r >= SOLVED_AVG)

    lines = [
        "=" * 55,
        "TRAINING SUMMARY — MountainCar PPO",
        "=" * 55,
        f"  Episodes trained         : {n:,}",
        f"  Policy updates           : {len(policy_losses)}",
        f"  Solved at episode        : {solve_ep if solve_ep else 'Not solved'}",
        f"  Mean reward (all)        : {np.mean(episode_raw_rewards):.2f} ± {np.std(episode_raw_rewards):.2f}",
        f"  Mean reward (last 100)   : {np.mean(last_100):.2f} ± {np.std(last_100):.2f}",
        f"  Success rate (last 100)  : {success} / {len(last_100)}  ({100*success/len(last_100):.1f}%)",
        f"  Mean steps (last 100)    : {np.mean(episode_lengths[-100:]):.1f}",
    ]
    if policy_losses:
        lines += [
            f"  Final policy loss        : {policy_losses[-1]:.6f}",
            f"  Final value loss         : {value_losses[-1]:.6f}",
            f"  Final entropy            : {entropies[-1]:.4f}",
        ]
    lines += [
        "",
        "  Hyperparameters",
        f"    Learning rate          : {LR}",
        f"    Gamma / Lambda (GAE)   : {GAMMA} / {LAM}",
        f"    Clip epsilon           : {CLIP_EPS}",
        f"    Entropy coef           : {ENTROPY_COEF}",
        f"    Value coef             : {VALUE_COEF}",
        f"    Epochs per update      : {EPOCHS}",
        f"    Batch size             : {BATCH_SIZE}",
        f"    Update freq (steps)    : {UPDATE_FREQ:,}",
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


def plot_results(episode_raw_rewards, episode_lengths, policy_losses,
                 value_losses, entropies, save_dir):
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
    if len(episode_lengths) >= 50:
        ma_len = np.convolve(episode_lengths, np.ones(50) / 50, mode='valid')
        axes[0, 2].plot(range(49, len(episode_lengths)), ma_len, color='saddlebrown', linewidth=2)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Steps')
    axes[0, 2].set_title('Steps per Episode')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Policy loss
    if policy_losses:
        axes[1, 0].plot(policy_losses, color='red', linewidth=1.5, marker='o', markersize=3)
    axes[1, 0].set_xlabel('Policy Update')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Policy Loss per Update')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Value loss + entropy (dual axis)
    ax5 = axes[1, 1]
    if value_losses:
        ax5.plot(value_losses, color='purple', linewidth=1.5, label='Value loss')
    ax5.set_xlabel('Policy Update')
    ax5.set_ylabel('Value Loss', color='purple')
    ax5.tick_params(axis='y', labelcolor='purple')
    ax5b = ax5.twinx()
    if entropies:
        ax5b.plot(entropies, color='green', linewidth=1.5, linestyle='--', label='Entropy')
    ax5b.set_ylabel('Entropy', color='green')
    ax5b.tick_params(axis='y', labelcolor='green')
    ax5.set_title('Value Loss & Entropy per Update')
    ax5.grid(True, alpha=0.3)

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

    plt.suptitle('MountainCar PPO — Training Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/training_results.png")


def plot_trajectory(agent, save_dir):
    """Run one greedy episode and plot position + velocity over time."""
    env = gym.make('MountainCar-v0')
    state, _ = env.reset(seed=0)
    positions, velocities = [], []

    for _ in range(MAX_STEPS):
        positions.append(state[0])
        velocities.append(state[1])
        action, _, _ = agent.select_action(state, deterministic=True)
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


def test_agent(agent, n_episodes=20):
    env = gym.make('MountainCar-v0')
    test_rewards, test_lengths = [], []

    print(f"\nTesting trained agent ({n_episodes} episodes)...")
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps        = 0

        for _ in range(MAX_STEPS):
            action, _, _ = agent.select_action(state, deterministic=True)
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
    results_dir = f"results/mountaincar_ppo_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    agent, raw_rewards, lengths, pl, vl, ent, solve_ep = train()

    save_log(raw_rewards, lengths, pl, vl, ent, results_dir)
    save_summary(raw_rewards, lengths, pl, vl, ent, solve_ep, results_dir)
    plot_results(raw_rewards, lengths, pl, vl, ent, results_dir)
    plot_trajectory(agent, results_dir)
    test_agent(agent)

    torch.save(agent.policy.state_dict(), f'{results_dir}/model.pth')
    print(f"\nResults saved to: {results_dir}/")
