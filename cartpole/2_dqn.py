import csv
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
EPS_START   = 1.0     # full exploration at start
EPS_END     = 0.05
EPS_DECAY   = 5000    # steps — slower decay prevents premature exploitation
TAU         = 0.005   # soft target update
LR          = 1e-4
MEMORY_SIZE = 50000   # larger buffer — prevents Q-value divergence from correlated data
N_EPISODES  = 1000
SOLVED_AVG  = 195


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


def select_action(policy_net, state, steps_done, n_actions):
    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-steps_done / EPS_DECAY)
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
        next_values[non_final_mask] = target_net(non_final_next).max(1)[0]

    expected = (next_values * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(state_action_values, expected.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss.item()


def soft_update(policy_net, target_net):
    for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
        tp.data.copy_(TAU * pp.data + (1 - TAU) * tp.data)


def train():
    env = gym.make('CartPole-v1')
    env.reset(seed=SEED)
    n_obs     = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = DQN(n_obs, n_actions).to(DEVICE)
    target_net = DQN(n_obs, n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory    = deque(maxlen=MEMORY_SIZE)

    episode_rewards = []
    loss_history    = []
    epsilon_history = []
    steps_done      = 0
    solve_ep        = None

    print(f"CartPole DQN | Device: {DEVICE}")
    print("=" * 50)

    for episode in range(N_EPISODES):
        state, _ = env.reset()
        state    = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        episode_losses = []

        for t in range(500):
            action, eps = select_action(policy_net, state, steps_done, n_actions)
            steps_done += 1

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done      = terminated or truncated
            reward_t  = torch.tensor([reward], device=DEVICE)
            next_state = None if terminated else torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            memory.append((state, action, next_state, reward_t))
            state = next_state

            loss = optimize(policy_net, target_net, optimizer, memory)
            if loss is not None:
                episode_losses.append(loss)

            soft_update(policy_net, target_net)
            if done:
                break

        episode_rewards.append(t + 1)
        epsilon_history.append(eps)
        loss_history.append(np.mean(episode_losses) if episode_losses else 0.0)

        if (episode + 1) % 50 == 0:
            avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"Episode {episode+1:4d} | Avg(100): {avg:6.2f} | Eps: {eps:.3f} | Loss: {loss_history[-1]:.4f}")

        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= SOLVED_AVG:
            if solve_ep is None:
                solve_ep = episode + 1
            print(f"\nSolved at episode {solve_ep}! Avg(100): {np.mean(episode_rewards[-100:]):.2f}")
            break

    env.close()
    return policy_net, episode_rewards, loss_history, epsilon_history, solve_ep


def save_log(episode_rewards, loss_history, epsilon_history, save_dir):
    path = f'{save_dir}/training_log.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'loss', 'epsilon'])
        for i in range(len(episode_rewards)):
            writer.writerow([i + 1, episode_rewards[i],
                             round(loss_history[i], 6), round(epsilon_history[i], 6)])
    print(f"Saved: {path}")


def save_summary(episode_rewards, loss_history, solve_ep, save_dir):
    n        = len(episode_rewards)
    last_100 = episode_rewards[-100:] if n >= 100 else episode_rewards
    success  = sum(1 for r in last_100 if r >= SOLVED_AVG)
    valid_losses = [l for l in loss_history if l > 0]

    lines = [
        "=" * 55,
        "TRAINING SUMMARY — CartPole DQN",
        "=" * 55,
        f"  Episodes trained         : {n:,}",
        f"  Solved at episode        : {solve_ep if solve_ep else 'Not solved'}",
        f"  Mean reward (all)        : {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}",
        f"  Mean reward (last 100)   : {np.mean(last_100):.2f} ± {np.std(last_100):.2f}",
        f"  Best single episode      : {int(np.max(episode_rewards))}",
        f"  Success rate (last 100)  : {success} / {len(last_100)}  ({100*success/len(last_100):.1f}%)",
        f"  Final mean loss          : {np.mean(loss_history[-50:]):.6f}",
        f"  Min mean loss            : {min(valid_losses):.6f}" if valid_losses else "  Min mean loss           : N/A",
        "",
        "  Hyperparameters",
        f"    Batch size             : {BATCH_SIZE}",
        f"    Gamma                  : {GAMMA}",
        f"    Learning rate          : {LR}",
        f"    Epsilon                : {EPS_START} → {EPS_END}  (decay steps: {EPS_DECAY})",
        f"    Soft update tau        : {TAU}",
        f"    Replay buffer size     : {MEMORY_SIZE:,}  (large buffer prevents Q-value divergence)",
        f"    Network                : 4 → 128 → 128 → 2  (ReLU, Huber loss, AdamW)",
        "=" * 55,
    ]
    text = '\n'.join(lines)
    print('\n' + text)
    with open(f'{save_dir}/summary.txt', 'w') as f:
        f.write(text + '\n')
    print(f"Saved: {save_dir}/summary.txt")


def plot_results(episode_rewards, loss_history, epsilon_history, save_dir):
    window = 50
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Raw rewards
    axes[0, 0].plot(episode_rewards, alpha=0.4, color='steelblue', linewidth=0.6, label='Per episode')
    axes[0, 0].axhline(y=SOLVED_AVG, color='red', linestyle='--', linewidth=1.5, label=f'Solved ({SOLVED_AVG})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward (= steps)')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Moving average
    if len(episode_rewards) >= window:
        ma = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        axes[0, 1].plot(range(window - 1, len(episode_rewards)), ma, color='green', linewidth=2)
        axes[0, 1].axhline(y=SOLVED_AVG, color='red', linestyle='--', linewidth=1.5, label=f'Solved ({SOLVED_AVG})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].set_title(f'Moving Average ({window} episodes)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Epsilon decay
    axes[0, 2].plot(epsilon_history, color='darkorange', linewidth=1.5)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Epsilon (ε)')
    axes[0, 2].set_title('Exploration Rate Decay')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Training loss
    valid_idx    = [i for i, l in enumerate(loss_history) if l > 0]
    valid_losses = [loss_history[i] for i in valid_idx]
    if valid_losses:
        axes[1, 0].plot(valid_idx, valid_losses, color='purple', alpha=0.6, linewidth=0.8)
        if len(valid_losses) >= 20:
            ma_loss = np.convolve(valid_losses, np.ones(20) / 20, mode='valid')
            axes[1, 0].plot([valid_idx[i] for i in range(19, len(valid_idx))],
                            ma_loss, color='black', linewidth=1.5, label='20-ep avg')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss (Huber)')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_yscale('symlog', linthresh=1e-4)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Reward distribution
    axes[1, 1].hist(episode_rewards, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    axes[1, 1].axvline(x=SOLVED_AVG, color='red', linestyle='--', linewidth=1.5, label=f'Solved ({SOLVED_AVG})')
    axes[1, 1].axvline(x=np.mean(episode_rewards), color='orange', linewidth=1.5,
                        label=f'Mean: {np.mean(episode_rewards):.1f}')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # 6. Summary statistics text
    n        = len(episode_rewards)
    last_100 = episode_rewards[-100:] if n >= 100 else episode_rewards
    axes[1, 2].axis('off')
    stats = (
        f"Episodes trained : {n}\n"
        f"Last-100 mean    : {np.mean(last_100):.1f}\n"
        f"Last-100 std     : {np.std(last_100):.1f}\n"
        f"Best episode     : {int(np.max(episode_rewards))}\n"
        f"Final epsilon    : {epsilon_history[-1]:.4f}\n\n"
        f"Config\n"
        f"  Batch size : {BATCH_SIZE}\n"
        f"  Gamma      : {GAMMA}\n"
        f"  LR         : {LR}\n"
        f"  Tau        : {TAU}\n"
        f"  Memory     : {MEMORY_SIZE:,}"
    )
    axes[1, 2].text(0.05, 0.95, stats, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    axes[1, 2].set_title('Run Statistics')

    plt.suptitle('CartPole DQN — Training Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/training_results.png")


def test_agent(policy_net, n_episodes=10):
    env = gym.make('CartPole-v1')
    test_rewards = []
    policy_net.eval()

    print(f"\nTesting trained agent ({n_episodes} episodes)...")
    for episode in range(n_episodes):
        state, _ = env.reset()
        state    = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        total_reward = 0

        for _ in range(500):
            with torch.no_grad():
                action = policy_net(state).max(1)[1].view(1, 1)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            state = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            total_reward += reward
            if terminated or truncated:
                break

        test_rewards.append(total_reward)
        status = "✓" if total_reward >= SOLVED_AVG else "✗"
        print(f"  {status} Test {episode+1:2d}: {total_reward:.0f} steps")

    avg = np.mean(test_rewards)
    print(f"\n  Average : {avg:.2f} ± {np.std(test_rewards):.2f}")
    print(f"  Result  : {'SOLVED' if avg >= SOLVED_AVG else 'Needs more training'}")
    env.close()


if __name__ == "__main__":
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/cartpole_dqn_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    policy_net, rewards, losses, epsilons, solve_ep = train()

    save_log(rewards, losses, epsilons, results_dir)
    save_summary(rewards, losses, solve_ep, results_dir)
    plot_results(rewards, losses, epsilons, results_dir)
    test_agent(policy_net)

    torch.save(policy_net.state_dict(), f'{results_dir}/model.pth')
    print(f"\nResults saved to: {results_dir}/")
