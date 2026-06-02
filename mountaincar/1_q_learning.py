import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import gymnasium as gym

SEED = 42
np.random.seed(SEED)

# Discretization
N_BINS       = [40, 40]
STATE_BOUNDS = [[-1.2, 0.6], [-0.07, 0.07]]

# Hyperparameters
LR         = 0.3
GAMMA      = 0.995
EPS_START  = 1.0
EPS_MIN    = 0.001
EPS_DECAY  = 0.9995
N_EPISODES = 25000
MAX_STEPS  = 200
SOLVED_AVG = -110.0

BINS = [np.linspace(lo, hi, n) for (lo, hi), n in zip(STATE_BOUNDS, N_BINS)]


def discretize(state):
    return tuple(
        np.clip(np.digitize(state[i], BINS[i]) - 1, 0, N_BINS[i] - 1)
        for i in range(2)
    )


def choose_action(q_table, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(3)
    return int(np.argmax(q_table[state]))


def train():
    env = gym.make('MountainCar-v0')
    env.reset(seed=SEED)
    q_table = np.zeros(N_BINS + [env.action_space.n])
    epsilon = EPS_START
    episode_rewards = []
    episode_lengths = []
    best_avg = -np.inf

    print("MountainCar Q-Learning")
    print("=" * 50)

    for episode in range(N_EPISODES):
        state, _ = env.reset()
        state = discretize(state)
        total_reward = 0
        steps = 0

        for _ in range(MAX_STEPS):
            action = choose_action(q_table, state, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize(next_obs)
            done = terminated or truncated

            current_q  = q_table[state + (action,)]
            # If terminated with reward > -200 the car reached the goal — no future value
            max_next_q = 0.0 if (done and reward > -200) else np.max(q_table[next_state])
            q_table[state + (action,)] += LR * (reward + GAMMA * max_next_q - current_q)

            state = next_state
            total_reward += reward
            steps += 1
            if done:
                break

        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if len(episode_rewards) >= 100:
            avg = np.mean(episode_rewards[-100:])
            if avg > best_avg:
                best_avg = avg

            if (episode + 1) % 500 == 0:
                success = sum(1 for r in episode_rewards[-100:] if r >= SOLVED_AVG)
                print(f"Ep {episode+1:5d} | Avg: {avg:7.2f} | Best: {best_avg:7.2f} | "
                      f"Success: {success:3d}/100 | ε: {epsilon:.4f}")

            if avg >= SOLVED_AVG:
                print(f"\nSolved at episode {episode+1}! Avg(100): {avg:.2f}")
                break

    env.close()
    return q_table, episode_rewards, episode_lengths


def plot_results(episode_rewards, episode_lengths, save_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    window = 100

    axes[0, 0].plot(episode_rewards, alpha=0.5, color='blue', linewidth=0.3)
    axes[0, 0].axhline(y=SOLVED_AVG, color='red', linestyle='--', linewidth=2, label=f'Solved ({SOLVED_AVG})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    if len(episode_rewards) >= window:
        ma = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        best_idx = np.argmax(ma)
        axes[0, 1].plot(range(window - 1, len(episode_rewards)), ma, color='green', linewidth=2)
        axes[0, 1].plot(best_idx + window - 1, ma[best_idx], 'r*', markersize=15, label=f'Best: {ma[best_idx]:.1f}')
        axes[0, 1].axhline(y=SOLVED_AVG, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].set_title(f'Moving Average ({window} episodes)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(episode_lengths, alpha=0.5, color='orange', linewidth=0.3)
    axes[0, 2].axhline(y=110, color='green', linestyle='--', linewidth=2, label='Target (<110 steps)')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Steps')
    axes[0, 2].set_title('Steps per Episode')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    if len(episode_lengths) >= window:
        ma_len = np.convolve(episode_lengths, np.ones(window) / window, mode='valid')
        axes[1, 0].plot(range(window - 1, len(episode_lengths)), ma_len, color='darkorange', linewidth=2)
        axes[1, 0].axhline(y=110, color='green', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Average Steps')
    axes[1, 0].set_title(f'Moving Average Steps ({window} episodes)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(episode_rewards, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=SOLVED_AVG, color='red', linestyle='--', linewidth=2, label=f'Solved ({SOLVED_AVG})')
    axes[1, 1].axvline(x=np.mean(episode_rewards), color='orange', linewidth=2,
                       label=f'Mean: {np.mean(episode_rewards):.1f}')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    success_rate = [
        100 * np.mean([1 if r >= SOLVED_AVG else 0
                       for r in episode_rewards[max(0, i - window):i + 1]])
        for i in range(len(episode_rewards))
    ]
    axes[1, 2].plot(success_rate, color='purple', linewidth=1.5)
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Success Rate (%)')
    axes[1, 2].set_title('Success Rate Over Time')
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle('MountainCar Q-Learning', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/training_results.png")


def test_agent(q_table, n_episodes=20):
    env = gym.make('MountainCar-v0')
    test_rewards, test_lengths = [], []

    print(f"\nTesting trained agent ({n_episodes} episodes, epsilon=0)...")
    for episode in range(n_episodes):
        state, _ = env.reset()
        state = discretize(state)
        total_reward = 0
        steps = 0

        for _ in range(MAX_STEPS):
            action = int(np.argmax(q_table[state]))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            state = discretize(next_obs)
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
    print(f"\n  Average: {avg:.2f}  |  Success: {success}/{n_episodes}")
    print(f"  {'Solved!' if avg >= SOLVED_AVG else 'Needs more training.'}")
    env.close()


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/mountaincar_qlearning_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    q_table, rewards, lengths = train()
    plot_results(rewards, lengths, results_dir)
    test_agent(q_table)

    np.save(f'{results_dir}/q_table.npy', q_table)
    print(f"\nResults saved to: {results_dir}/")
