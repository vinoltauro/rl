import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import gymnasium as gym

SEED = 42
np.random.seed(SEED)

PLT_STYLE = {
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.titlesize': 14,
}
plt.rcParams.update(PLT_STYLE)

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
    q_table         = np.zeros(N_BINS + [2])
    epsilon         = EPS_START
    episode_rewards = []
    epsilon_history = []
    solve_ep        = None

    print("CartPole Q-Learning")
    print("=" * 50)

    for episode in range(N_EPISODES):
        state, _ = env.reset()
        state    = discretize(state)
        total_reward = 0

        for _ in range(MAX_STEPS):
            action = choose_action(q_table, state, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize(next_obs)
            done       = terminated or truncated

            current_q  = q_table[state + (action,)]
            max_next_q = 0.0 if done else np.max(q_table[next_state])
            q_table[state + (action,)] += LR * (reward + GAMMA * max_next_q - current_q)

            state = next_state
            total_reward += reward
            if done:
                break

        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)
        episode_rewards.append(total_reward)
        epsilon_history.append(epsilon)

        if (episode + 1) % 500 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1:5d} | Avg(100): {avg:6.2f} | Epsilon: {epsilon:.3f}")

        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= SOLVED_AVG:
            if solve_ep is None:
                solve_ep = episode + 1
            print(f"\nSolved at episode {solve_ep}! Avg(100): {np.mean(episode_rewards[-100:]):.2f}")
            break

    env.close()
    return q_table, episode_rewards, epsilon_history, solve_ep


def save_log(episode_rewards, epsilon_history, save_dir):
    path = f'{save_dir}/training_log.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'epsilon'])
        for i in range(len(episode_rewards)):
            writer.writerow([i + 1, episode_rewards[i], round(epsilon_history[i], 6)])
    print(f"Saved: {path}")


def save_summary(episode_rewards, solve_ep, save_dir):
    n        = len(episode_rewards)
    last_100 = episode_rewards[-100:] if n >= 100 else episode_rewards
    success  = sum(1 for r in last_100 if r >= SOLVED_AVG)

    lines = [
        "=" * 55,
        "TRAINING SUMMARY — CartPole Q-Learning",
        "=" * 55,
        f"  Episodes trained         : {n:,}",
        f"  Solved at episode        : {solve_ep if solve_ep else 'Not solved'}",
        f"  Mean reward (all)        : {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}",
        f"  Mean reward (last 100)   : {np.mean(last_100):.2f} ± {np.std(last_100):.2f}",
        f"  Best single episode      : {int(np.max(episode_rewards))}",
        f"  Success rate (last 100)  : {success} / {len(last_100)}  ({100*success/len(last_100):.1f}%)",
        "",
        "  Hyperparameters",
        f"    Alpha (LR)             : {LR}",
        f"    Gamma                  : {GAMMA}",
        f"    Epsilon start → min    : {EPS_START} → {EPS_MIN}  (decay {EPS_DECAY})",
        f"    Bins per dim           : {N_BINS}  →  {int(np.prod(N_BINS)):,} states",
        f"    Solved threshold       : avg ≥ {SOLVED_AVG} over 100 episodes",
        "=" * 55,
    ]
    text = '\n'.join(lines)
    print('\n' + text)
    with open(f'{save_dir}/summary.txt', 'w') as f:
        f.write(text + '\n')
    print(f"Saved: {save_dir}/summary.txt")


def plot_results(episode_rewards, epsilon_history, save_dir):
    window = 100
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Raw rewards
    axes[0, 0].plot(episode_rewards, alpha=0.4, color='steelblue', linewidth=0.6, label='Per episode')
    if len(episode_rewards) >= window:
        ma = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        axes[0, 0].plot(range(window - 1, len(episode_rewards)), ma, color='navy', linewidth=2, label=f'{window}-ep avg')
    axes[0, 0].axhline(y=SOLVED_AVG, color='red', linestyle='--', linewidth=1.5, label=f'Solved ({SOLVED_AVG})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward (= steps balanced)')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Moving average
    if len(episode_rewards) >= window:
        axes[0, 1].plot(range(window - 1, len(episode_rewards)), ma, color='green', linewidth=2, label=f'{window}-ep avg')
        axes[0, 1].axhline(y=SOLVED_AVG, color='red', linestyle='--', linewidth=1.5, label=f'Solved ({SOLVED_AVG})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].set_title(f'Moving Average ({window} episodes)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Epsilon decay
    axes[1, 0].plot(epsilon_history, color='darkorange', linewidth=1.5)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon (ε)')
    axes[1, 0].set_title('Exploration Rate Decay')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Reward distribution
    axes[1, 1].hist(episode_rewards, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    axes[1, 1].axvline(x=SOLVED_AVG, color='red', linestyle='--', linewidth=1.5, label=f'Solved ({SOLVED_AVG})')
    axes[1, 1].axvline(x=np.mean(episode_rewards), color='orange', linestyle='-', linewidth=1.5,
                        label=f'Mean: {np.mean(episode_rewards):.1f}')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('CartPole Q-Learning — Training Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/training_results.png")


def test_agent(q_table, n_episodes=10):
    env = gym.make('CartPole-v1')
    test_rewards = []

    print(f"\nTesting trained agent ({n_episodes} episodes, ε=0)...")
    for episode in range(n_episodes):
        state, _ = env.reset()
        state    = discretize(state)
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
    print(f"\n  Average : {avg:.2f} ± {np.std(test_rewards):.2f}")
    print(f"  Result  : {'SOLVED' if avg >= SOLVED_AVG else 'Needs more training'}")
    env.close()


if __name__ == "__main__":
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/cartpole_qlearning_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    q_table, rewards, epsilons, solve_ep = train()

    save_log(rewards, epsilons, results_dir)
    save_summary(rewards, solve_ep, results_dir)
    plot_results(rewards, epsilons, results_dir)
    test_agent(q_table)

    np.save(f'{results_dir}/q_table.npy', q_table)
    print(f"\nResults saved to: {results_dir}/")
