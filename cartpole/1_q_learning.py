import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import gymnasium as gym

SEED = 42
np.random.seed(SEED)

# Discretization
N_BINS       = [10, 10, 20, 20]
STATE_BOUNDS = [[-4.8, 4.8], [-4.0, 4.0], [-0.418, 0.418], [-4.0, 4.0]]

# Hyperparameters
LR         = 0.1
GAMMA      = 0.99
EPS_START  = 1.0
EPS_MIN    = 0.01
EPS_DECAY  = 0.995
N_EPISODES = 10000
MAX_STEPS  = 500
SOLVED_AVG = 195

BINS = [np.linspace(lo, hi, n) for (lo, hi), n in zip(STATE_BOUNDS, N_BINS)]


def discretize(state):
    return tuple(
        np.clip(np.digitize(state[i], BINS[i]) - 1, 0, N_BINS[i] - 1)
        for i in range(4)
    )


def choose_action(q_table, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(2)
    return int(np.argmax(q_table[state]))


def train():
    env = gym.make('CartPole-v1')
    env.reset(seed=SEED)
    q_table = np.zeros(N_BINS + [2])
    epsilon = EPS_START
    episode_rewards = []

    print("CartPole Q-Learning")
    print("=" * 50)

    for episode in range(N_EPISODES):
        state, _ = env.reset()
        state = discretize(state)
        total_reward = 0

        for _ in range(MAX_STEPS):
            action = choose_action(q_table, state, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize(next_obs)
            done = terminated or truncated

            current_q  = q_table[state + (action,)]
            max_next_q = 0.0 if done else np.max(q_table[next_state])
            q_table[state + (action,)] += LR * (reward + GAMMA * max_next_q - current_q)

            state = next_state
            total_reward += reward
            if done:
                break

        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)
        episode_rewards.append(total_reward)

        if (episode + 1) % 500 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1:5d} | Avg(100): {avg:6.2f} | Epsilon: {epsilon:.3f}")

        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= SOLVED_AVG:
            print(f"\nSolved at episode {episode+1}! Avg(100): {np.mean(episode_rewards[-100:]):.2f}")
            break

    env.close()
    return q_table, episode_rewards


def plot_results(episode_rewards, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(episode_rewards, alpha=0.5, color='blue', linewidth=0.5)
    axes[0].axhline(y=SOLVED_AVG, color='red', linestyle='--', label=f'Solved ({SOLVED_AVG})')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Episode Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    window = 100
    if len(episode_rewards) >= window:
        ma = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        axes[1].plot(range(window - 1, len(episode_rewards)), ma, color='green', linewidth=2)
        axes[1].axhline(y=SOLVED_AVG, color='red', linestyle='--', label=f'Solved ({SOLVED_AVG})')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Average Reward')
    axes[1].set_title(f'Moving Average ({window} episodes)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(episode_rewards, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
    axes[2].axvline(x=SOLVED_AVG, color='red', linestyle='--', label=f'Solved ({SOLVED_AVG})')
    axes[2].set_xlabel('Reward')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Reward Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('CartPole Q-Learning', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/training_results.png")


def test_agent(q_table, n_episodes=10):
    env = gym.make('CartPole-v1')
    test_rewards = []

    print(f"\nTesting trained agent ({n_episodes} episodes, epsilon=0)...")
    for episode in range(n_episodes):
        state, _ = env.reset()
        state = discretize(state)
        total_reward = 0

        for _ in range(MAX_STEPS):
            action = int(np.argmax(q_table[state]))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            state = discretize(next_obs)
            total_reward += reward
            if terminated or truncated:
                break

        test_rewards.append(total_reward)
        status = "✓" if total_reward >= SOLVED_AVG else "✗"
        print(f"  {status} Test {episode+1:2d}: {total_reward:.0f} steps")

    avg = np.mean(test_rewards)
    print(f"\n  Average: {avg:.2f}")
    print(f"  {'Solved!' if avg >= SOLVED_AVG else 'Needs more training.'}")
    env.close()


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/cartpole_qlearning_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    q_table, rewards = train()
    plot_results(rewards, results_dir)
    test_agent(q_table)

    np.save(f'{results_dir}/q_table.npy', q_table)
    print(f"\nResults saved to: {results_dir}/")
