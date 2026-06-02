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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE   = 64
GAMMA        = 0.99
EPS_START    = 1.0
EPS_END      = 0.01
EPS_DECAY    = 1000   # steps
TAU          = 0.005  # soft target update
LR           = 1e-3
MEMORY_SIZE  = 20000
N_EPISODES   = 600
MAX_STEPS    = 400    # slightly extended to help exploration
SOLVED_AVG   = -110.0


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


def shape_reward(pos, vel, raw_reward, terminated):
    """Physics-based reward shaping: height + kinetic energy bonus."""
    height_bonus = abs(pos + 0.5) * 10.0
    shaped = height_bonus
    if pos >= 0.5:
        shaped += 100.0
    return shaped


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
        next_values[non_final_mask] = target_net(non_final_next).max(1)[0]

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

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory    = deque(maxlen=MEMORY_SIZE)

    episode_rewards = []
    loss_history    = []
    epsilon_history = []
    steps_done      = 0

    print(f"MountainCar DQN (reward shaping) | Device: {DEVICE}")
    print("=" * 50)

    for episode in range(N_EPISODES):
        state, _ = env.reset()
        state_t   = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        total_reward  = 0
        episode_losses = []

        for _ in range(MAX_STEPS):
            action, eps = select_action(policy_net, state_t, steps_done, n_actions)
            steps_done  += 1

            next_obs, raw_reward, terminated, truncated, _ = env.step(action.item())
            done    = terminated or truncated
            shaped  = shape_reward(next_obs[0], next_obs[1], raw_reward, terminated)
            total_reward += raw_reward  # track real reward for evaluation

            shaped_t   = torch.tensor([shaped], dtype=torch.float32, device=DEVICE)
            next_state = None if terminated else torch.tensor(next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            memory.append((state_t, action, next_state, shaped_t))
            state_t = next_state

            loss = optimize(policy_net, target_net, optimizer, memory)
            if loss is not None:
                episode_losses.append(loss)

            soft_update(policy_net, target_net)
            if done:
                break

        episode_rewards.append(total_reward)
        epsilon_history.append(eps)
        loss_history.append(np.mean(episode_losses) if episode_losses else 0.0)

        if (episode + 1) % 50 == 0:
            avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"Episode {episode+1:4d} | Avg(100): {avg:7.2f} | Eps: {eps:.3f}")

        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= SOLVED_AVG:
            print(f"\nSolved at episode {episode+1}! Avg(100): {np.mean(episode_rewards[-100:]):.2f}")
            break

    env.close()
    return policy_net, episode_rewards, loss_history, epsilon_history


def plot_results(episode_rewards, loss_history, epsilon_history, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(episode_rewards, alpha=0.5, color='blue', linewidth=0.5)
    axes[0, 0].axhline(y=SOLVED_AVG, color='red', linestyle='--', label=f'Solved ({SOLVED_AVG})')
    axes[0, 0].set_title('Episode Rewards (real score)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    window = 50
    if len(episode_rewards) >= window:
        ma = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        axes[0, 1].plot(range(window - 1, len(episode_rewards)), ma, color='green', linewidth=2)
        axes[0, 1].axhline(y=SOLVED_AVG, color='red', linestyle='--', label=f'Solved ({SOLVED_AVG})')
    axes[0, 1].set_title(f'Moving Average ({window} episodes)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(loss_history, color='purple', alpha=0.7, linewidth=0.8)
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epsilon_history, color='orange', linewidth=2)
    axes[1, 1].set_title('Exploration Rate (Epsilon)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('MountainCar DQN (reward shaping)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/training_results.png")


def test_agent(policy_net, n_episodes=20):
    env = gym.make('MountainCar-v0')
    test_rewards = []
    policy_net.eval()

    print(f"\nTesting trained agent ({n_episodes} episodes)...")
    for episode in range(n_episodes):
        state, _ = env.reset()
        state_t   = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        total_reward = 0

        for _ in range(200):
            with torch.no_grad():
                action = policy_net(state_t).max(1)[1].view(1, 1)
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            state_t = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            total_reward += reward
            if terminated or truncated:
                break

        test_rewards.append(total_reward)
        status = "✓" if total_reward >= SOLVED_AVG else "✗"
        print(f"  {status} Test {episode+1:2d}: {total_reward:6.1f}")

    avg     = np.mean(test_rewards)
    success = sum(1 for r in test_rewards if r >= SOLVED_AVG)
    print(f"\n  Average: {avg:.2f}  |  Success: {success}/{n_episodes}")
    print(f"  {'Solved!' if avg >= SOLVED_AVG else 'Needs more training.'}")
    env.close()


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/mountaincar_dqn_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    policy_net, rewards, losses, epsilons = train()
    plot_results(rewards, losses, epsilons, results_dir)
    test_agent(policy_net)

    torch.save(policy_net.state_dict(), f'{results_dir}/model.pth')
    print(f"\nResults saved to: {results_dir}/")
