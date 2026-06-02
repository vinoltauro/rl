import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from datetime import datetime
import os
import gymnasium as gym

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PLT_STYLE = {
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.titlesize': 14,
}
plt.rcParams.update(PLT_STYLE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE  = 128
GAMMA       = 0.99
EPS_START   = 1.0
EPS_END     = 0.05    # floor at 5%: prevents fully deterministic policy getting stuck in local attractor
EPS_DECAY   = 50000   # steps — with 200 steps/ep, stays meaningful for ~400+ episodes
TAU         = 0.005   # soft target update
LR          = 5e-5    # lowered from 1e-4; Double DQN still benefits from smaller LR
MEMORY_SIZE = 50000
N_EPISODES  = 5000
MAX_STEPS   = 200     # standardised to match AC/PPO and gym default
SOLVED_AVG  = -110.0


class DQN(nn.Module):
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 128), nn.ReLU(),
            nn.Linear(128, 128),   nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def shape_reward(pos, vel, terminated):
    """Physics-based reward shaping: height + kinetic energy + step penalty."""
    height = (pos + 1.2) / 1.8 * 2.0   # maps [-1.2, 0.6] → [0, 2]
    ke     = vel * vel                   # kinetic energy proxy
    reward = height + 100.0 * ke - 1.0  # step penalty encourages efficiency
    if terminated and pos >= 0.5:
        reward += 10.0   # align with AC/PPO; +100 inflated Q-targets and caused divergence
    return reward


def select_action(policy_net, state, steps_done, n_actions):
    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done / EPS_DECAY)
    if random.random() > eps:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1), eps
    return torch.tensor([[random.randrange(n_actions)]], device=DEVICE, dtype=torch.long), eps


def optimize(policy_net, target_net, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return None

    transitions  = random.sample(memory, BATCH_SIZE)
    states, actions, next_states, rewards = zip(*transitions)

    state_batch  = torch.cat(states)
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    non_final_mask = torch.tensor([s is not None for s in next_states], device=DEVICE, dtype=torch.bool)
    non_final_next = torch.cat([s for s in next_states if s is not None])
    next_values    = torch.zeros(BATCH_SIZE, device=DEVICE)
    with torch.no_grad():
        # Double DQN: policy_net selects action, target_net evaluates it.
        # Decouples selection from evaluation → removes overestimation bias
        # that caused Q-values to diverge (loss ~2500) in standard DQN.
        next_actions = policy_net(non_final_next).max(1)[1].unsqueeze(1)
        next_values[non_final_mask] = target_net(non_final_next).gather(1, next_actions).squeeze(1)

    expected = (next_values * GAMMA) + reward_batch
    loss     = F.smooth_l1_loss(state_action_values, expected.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def soft_update(policy_net, target_net):
    for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
        tp.data.copy_(TAU * pp.data + (1 - TAU) * tp.data)


def train():
    env = gym.make('MountainCar-v0', max_episode_steps=MAX_STEPS)
    env.reset(seed=SEED)
    n_obs     = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = DQN(n_obs, n_actions).to(DEVICE)
    target_net = DQN(n_obs, n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory    = deque(maxlen=MEMORY_SIZE)

    episode_rewards = []
    episode_lengths = []
    loss_history    = []
    epsilon_history = []
    steps_done      = 0
    solve_ep        = None

    print(f"MountainCar DQN (reward shaping) | Device: {DEVICE}")
    print("=" * 55)

    for episode in range(N_EPISODES):
        state, _ = env.reset()
        state_t  = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        total_reward   = 0
        steps          = 0
        episode_losses = []

        for _ in range(MAX_STEPS):
            action, eps = select_action(policy_net, state_t, steps_done, n_actions)
            steps_done += 1

            next_obs, raw_reward, terminated, truncated, _ = env.step(action.item())
            done    = terminated or truncated
            shaped  = shape_reward(next_obs[0], next_obs[1], terminated)
            total_reward += raw_reward   # always track real reward

            shaped_t   = torch.tensor([shaped], dtype=torch.float32, device=DEVICE)
            next_state = None if terminated else torch.tensor(next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            memory.append((state_t, action, next_state, shaped_t))
            state_t = next_state
            steps  += 1

            loss = optimize(policy_net, target_net, optimizer, memory)
            if loss is not None:
                episode_losses.append(loss)

            soft_update(policy_net, target_net)
            if done:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        epsilon_history.append(eps)
        loss_history.append(np.mean(episode_losses) if episode_losses else 0.0)

        if (episode + 1) % 50 == 0:
            avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"Episode {episode+1:4d} | Avg(100): {avg:7.2f} | Eps: {eps:.4f} | Loss: {loss_history[-1]:.4f}")

        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= SOLVED_AVG:
            if solve_ep is None:
                solve_ep = episode + 1
            print(f"\nSolved at episode {solve_ep}! Avg(100): {np.mean(episode_rewards[-100:]):.2f}")
            break

    env.close()
    return policy_net, episode_rewards, episode_lengths, loss_history, epsilon_history, solve_ep


def save_log(episode_rewards, episode_lengths, loss_history, epsilon_history, save_dir):
    path = f'{save_dir}/training_log.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'steps', 'loss', 'epsilon'])
        for i in range(len(episode_rewards)):
            writer.writerow([i + 1, episode_rewards[i], episode_lengths[i],
                             round(loss_history[i], 6), round(epsilon_history[i], 6)])
    print(f"Saved: {path}")


def save_summary(episode_rewards, episode_lengths, loss_history, solve_ep, save_dir):
    n        = len(episode_rewards)
    last_100 = episode_rewards[-100:] if n >= 100 else episode_rewards
    success  = sum(1 for r in last_100 if r >= SOLVED_AVG)
    valid_l  = [l for l in loss_history if l > 0]

    lines = [
        "=" * 55,
        "TRAINING SUMMARY — MountainCar DQN",
        "=" * 55,
        f"  Episodes trained         : {n:,}",
        f"  Solved at episode        : {solve_ep if solve_ep else 'Not solved'}",
        f"  Mean reward (all)        : {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}",
        f"  Mean reward (last 100)   : {np.mean(last_100):.2f} ± {np.std(last_100):.2f}",
        f"  Success rate (last 100)  : {success} / {len(last_100)}  ({100*success/len(last_100):.1f}%)",
        f"  Mean steps (last 100)    : {np.mean(episode_lengths[-100:]):.1f}",
        f"  Final mean loss          : {np.mean(loss_history[-50:]):.6f}",
        "",
        "  Hyperparameters",
        f"    Batch size             : {BATCH_SIZE}",
        f"    Gamma                  : {GAMMA}",
        f"    Learning rate          : {LR}",
        f"    Epsilon                : {EPS_START} → {EPS_END}  (decay steps: {EPS_DECAY})",
        f"    Soft update tau        : {TAU}",
        f"    Replay buffer          : {MEMORY_SIZE:,}",
        f"    Max steps/episode      : {MAX_STEPS}",
        f"    Reward shaping         : height + 100·KE - 1  (+10 at goal)",
        "=" * 55,
    ]
    text = '\n'.join(lines)
    print('\n' + text)
    with open(f'{save_dir}/summary.txt', 'w') as f:
        f.write(text + '\n')
    print(f"Saved: {save_dir}/summary.txt")


def plot_results(episode_rewards, episode_lengths, loss_history, epsilon_history, save_dir):
    window = 50
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Raw rewards
    axes[0, 0].plot(episode_rewards, alpha=0.4, color='steelblue', linewidth=0.6, label='Per episode')
    axes[0, 0].axhline(y=SOLVED_AVG, color='red', linestyle='--', linewidth=1.5, label=f'Solved ({SOLVED_AVG})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward (real score)')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Moving average
    if len(episode_rewards) >= window:
        ma = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        axes[0, 1].plot(range(window - 1, len(episode_rewards)), ma, color='green', linewidth=2, label=f'{window}-ep avg')
        axes[0, 1].axhline(y=SOLVED_AVG, color='red', linestyle='--', linewidth=1.5)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].set_title(f'Moving Average ({window} episodes)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Steps per episode
    axes[0, 2].plot(episode_lengths, alpha=0.4, color='darkorange', linewidth=0.6)
    axes[0, 2].axhline(y=110, color='green', linestyle='--', linewidth=1.5, label='Target (≤110)')
    if len(episode_lengths) >= window:
        ma_len = np.convolve(episode_lengths, np.ones(window) / window, mode='valid')
        axes[0, 2].plot(range(window - 1, len(episode_lengths)), ma_len, color='saddlebrown', linewidth=2)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Steps')
    axes[0, 2].set_title('Steps per Episode')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Training loss
    valid_idx    = [i for i, l in enumerate(loss_history) if l > 0]
    valid_losses = [loss_history[i] for i in valid_idx]
    if valid_losses:
        axes[1, 0].plot(valid_idx, valid_losses, color='purple', alpha=0.5, linewidth=0.8)
        if len(valid_losses) >= 20:
            ma_loss = np.convolve(valid_losses, np.ones(20) / 20, mode='valid')
            axes[1, 0].plot([valid_idx[i] for i in range(19, len(valid_idx))],
                            ma_loss, color='black', linewidth=1.5, label='20-ep trend')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss (Huber)')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_yscale('symlog', linthresh=1e-4)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Epsilon decay
    axes[1, 1].plot(epsilon_history, color='darkorange', linewidth=1.5)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon (ε)')
    axes[1, 1].set_title('Exploration Rate Decay')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Reward distribution
    axes[1, 2].hist(episode_rewards, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    axes[1, 2].axvline(x=SOLVED_AVG, color='red', linestyle='--', linewidth=1.5, label=f'Solved ({SOLVED_AVG})')
    axes[1, 2].axvline(x=np.mean(episode_rewards), color='orange', linewidth=1.5,
                        label=f'Mean: {np.mean(episode_rewards):.1f}')
    axes[1, 2].set_xlabel('Reward')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Reward Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('MountainCar DQN — Training Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/training_results.png")


def plot_trajectory(policy_net, save_dir):
    """Run one greedy episode and plot position + velocity over time."""
    env = gym.make('MountainCar-v0')
    state, _ = env.reset(seed=0)
    positions, velocities = [], []
    policy_net.eval()

    for _ in range(200):
        positions.append(state[0])
        velocities.append(state[1])
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action = policy_net(state_t).max(1)[1].item()
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


def test_agent(policy_net, n_episodes=20):
    env = gym.make('MountainCar-v0')
    test_rewards, test_lengths = [], []
    policy_net.eval()

    print(f"\nTesting trained agent ({n_episodes} episodes)...")
    for episode in range(n_episodes):
        state, _ = env.reset()
        state_t  = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        total_reward = 0
        steps = 0

        for _ in range(200):
            with torch.no_grad():
                action = policy_net(state_t).max(1)[1].item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            state_t = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
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
    results_dir = f"results/mountaincar_dqn_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    policy_net, rewards, lengths, losses, epsilons, solve_ep = train()

    save_log(rewards, lengths, losses, epsilons, results_dir)
    save_summary(rewards, lengths, losses, solve_ep, results_dir)
    plot_results(rewards, lengths, losses, epsilons, results_dir)
    plot_trajectory(policy_net, results_dir)
    test_agent(policy_net)

    torch.save(policy_net.state_dict(), f'{results_dir}/model.pth')
    print(f"\nResults saved to: {results_dir}/")
