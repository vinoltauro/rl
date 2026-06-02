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
N_BINS       = [40, 40]
STATE_BOUNDS = [[-1.2, 0.6], [-0.07, 0.07]]

# Hyperparameters
LR         = 0.3
GAMMA      = 0.995
EPS_START  = 1.0
EPS_MIN    = 0.001
EPS_DECAY  = 0.9997   # slower decay — hits EPS_MIN at ~ep 11K, better exploration budget
N_EPISODES = 50000
MAX_STEPS  = 200
SOLVED_AVG = -110.0

BINS = [np.linspace(lo, hi, n) for (lo, hi), n in zip(STATE_BOUNDS, N_BINS)]


def shape_reward(pos, vel, terminated):
    """Same shaping used by DQN/AC/PPO — ensures fair comparison across algorithms."""
    height = (pos + 1.2) / 1.8 * 2.0
    ke     = vel * vel
    reward = height + 100.0 * ke - 1.0
    if terminated and pos >= 0.5:
        reward += 10.0
    return reward


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
    q_table         = np.zeros(N_BINS + [env.action_space.n])
    epsilon         = EPS_START
    episode_rewards = []
    episode_lengths = []
    epsilon_history = []
    best_avg        = -np.inf
    solve_ep        = None

    print("MountainCar Q-Learning")
    print("=" * 55)

    for episode in range(N_EPISODES):
        state, _ = env.reset()
        state    = discretize(state)
        total_reward = 0
        steps        = 0

        for _ in range(MAX_STEPS):
            action = choose_action(q_table, state, epsilon)
            next_obs, raw_reward, terminated, truncated, _ = env.step(action)
            next_state  = discretize(next_obs)
            done        = terminated or truncated
            shaped      = shape_reward(next_obs[0], next_obs[1], terminated)

            current_q  = q_table[state + (action,)]
            max_next_q = 0.0 if terminated else np.max(q_table[next_state])
            q_table[state + (action,)] += LR * (shaped + GAMMA * max_next_q - current_q)

            state = next_state
            total_reward += raw_reward   # track real score for solved check
            steps += 1
            if done:
                break

        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        epsilon_history.append(epsilon)

        if len(episode_rewards) >= 100:
            avg = np.mean(episode_rewards[-100:])
            if avg > best_avg:
                best_avg = avg

            if (episode + 1) % 500 == 0:
                success = sum(1 for r in episode_rewards[-100:] if r >= SOLVED_AVG)
                print(f"Ep {episode+1:5d} | Avg: {avg:7.2f} | Best: {best_avg:7.2f} | "
                      f"Success: {success:3d}/100 | ε: {epsilon:.4f}")

            if avg >= SOLVED_AVG:
                if solve_ep is None:
                    solve_ep = episode + 1
                print(f"\nSolved at episode {solve_ep}! Avg(100): {avg:.2f}")
                break

    env.close()
    return q_table, episode_rewards, episode_lengths, epsilon_history, solve_ep


def save_log(episode_rewards, episode_lengths, epsilon_history, save_dir):
    path = f'{save_dir}/training_log.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'steps', 'epsilon'])
        for i in range(len(episode_rewards)):
            writer.writerow([i + 1, episode_rewards[i], episode_lengths[i],
                             round(epsilon_history[i], 6)])
    print(f"Saved: {path}")


def save_summary(episode_rewards, episode_lengths, solve_ep, save_dir):
    n        = len(episode_rewards)
    last_100 = episode_rewards[-100:] if n >= 100 else episode_rewards
    success  = sum(1 for r in last_100 if r >= SOLVED_AVG)

    lines = [
        "=" * 55,
        "TRAINING SUMMARY — MountainCar Q-Learning",
        "=" * 55,
        f"  Episodes trained         : {n:,}",
        f"  Solved at episode        : {solve_ep if solve_ep else 'Not solved'}",
        f"  Mean reward (all)        : {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}",
        f"  Mean reward (last 100)   : {np.mean(last_100):.2f} ± {np.std(last_100):.2f}",
        f"  Best 100-ep average      : {max(np.mean(episode_rewards[i:i+100]) for i in range(max(1, n-100))):.2f}",
        f"  Success rate (last 100)  : {success} / {len(last_100)}  ({100*success/len(last_100):.1f}%)",
        f"  Mean steps (last 100)    : {np.mean(episode_lengths[-100:]):.1f}",
        "",
        "  Hyperparameters",
        f"    Alpha (LR)             : {LR}",
        f"    Gamma                  : {GAMMA}",
        f"    Epsilon start → min    : {EPS_START} → {EPS_MIN}  (decay {EPS_DECAY})",
        f"    Bins (pos × vel)       : {N_BINS[0]} × {N_BINS[1]} = {N_BINS[0]*N_BINS[1]:,} states",
        f"    Reward shaping         : height + 100·KE - 1  (+10 at goal)",
        f"    Solved threshold       : avg ≥ {SOLVED_AVG} over 100 episodes",
        "=" * 55,
    ]
    text = '\n'.join(lines)
    print('\n' + text)
    with open(f'{save_dir}/summary.txt', 'w') as f:
        f.write(text + '\n')
    print(f"Saved: {save_dir}/summary.txt")


def plot_results(episode_rewards, episode_lengths, save_dir):
    window = 100
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Raw rewards
    axes[0, 0].plot(episode_rewards, alpha=0.4, color='steelblue', linewidth=0.3, label='Per episode')
    axes[0, 0].axhline(y=SOLVED_AVG, color='red', linestyle='--', linewidth=1.5, label=f'Solved ({SOLVED_AVG})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Moving average
    if len(episode_rewards) >= window:
        ma      = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        best_idx = np.argmax(ma)
        axes[0, 1].plot(range(window - 1, len(episode_rewards)), ma, color='green', linewidth=2, label=f'{window}-ep avg')
        axes[0, 1].plot(best_idx + window - 1, ma[best_idx], 'r*', markersize=15, label=f'Best: {ma[best_idx]:.1f}')
        axes[0, 1].axhline(y=SOLVED_AVG, color='red', linestyle='--', linewidth=1.5)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].set_title(f'Moving Average ({window} episodes)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Steps per episode
    axes[0, 2].plot(episode_lengths, alpha=0.4, color='darkorange', linewidth=0.3, label='Per episode')
    axes[0, 2].axhline(y=110, color='green', linestyle='--', linewidth=1.5, label='Target (≤110 steps)')
    if len(episode_lengths) >= window:
        ma_len = np.convolve(episode_lengths, np.ones(window) / window, mode='valid')
        axes[0, 2].plot(range(window - 1, len(episode_lengths)), ma_len, color='saddlebrown', linewidth=2)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Steps')
    axes[0, 2].set_title('Steps per Episode')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Success rate over time
    success_rate = [
        100 * np.mean([1 if r >= SOLVED_AVG else 0
                       for r in episode_rewards[max(0, i - window):i + 1]])
        for i in range(len(episode_rewards))
    ]
    axes[1, 0].plot(success_rate, color='purple', linewidth=1.2)
    axes[1, 0].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate (%)')
    axes[1, 0].set_title(f'Success Rate (rolling {window}-ep window)')
    axes[1, 0].set_ylim(0, 105)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Reward distribution
    axes[1, 1].hist(episode_rewards, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
    axes[1, 1].axvline(x=SOLVED_AVG, color='red', linestyle='--', linewidth=1.5, label=f'Solved ({SOLVED_AVG})')
    axes[1, 1].axvline(x=np.mean(episode_rewards), color='orange', linewidth=1.5,
                        label=f'Mean: {np.mean(episode_rewards):.1f}')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # 6. Steps distribution (last 100 episodes)
    last_lens = episode_lengths[-100:]
    axes[1, 2].hist(last_lens, bins=20, color='darkorange', edgecolor='white', alpha=0.85)
    axes[1, 2].axvline(x=110, color='green', linestyle='--', linewidth=1.5, label='Target (110)')
    axes[1, 2].axvline(x=np.mean(last_lens), color='red', linewidth=1.5,
                        label=f'Mean: {np.mean(last_lens):.1f}')
    axes[1, 2].set_xlabel('Steps')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Steps Distribution (last 100 episodes)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('MountainCar Q-Learning — Training Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/training_results.png")


def plot_trajectory(q_table, save_dir):
    """Run one greedy episode and plot position + velocity over time."""
    env = gym.make('MountainCar-v0')
    state, _ = env.reset(seed=0)
    positions, velocities = [], []

    for _ in range(MAX_STEPS):
        positions.append(state[0])
        velocities.append(state[1])
        action = int(np.argmax(q_table[discretize(state)]))
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    positions.append(state[0])
    velocities.append(state[1])
    env.close()

    reached = max(positions) >= 0.5
    status  = f"REACHED GOAL in {len(positions)-1} steps" if reached else f"DID NOT REACH GOAL ({len(positions)-1} steps)"
    timesteps = range(len(positions))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(timesteps, positions, color='steelblue', linewidth=1.8)
    axes[0].axhline(y=0.5, color='green', linestyle='--', linewidth=1.5, label='Goal (pos = 0.5)')
    axes[0].axhline(y=-0.5, color='gray', linestyle=':', alpha=0.5, label='Valley bottom (-0.5)')
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


def test_agent(q_table, n_episodes=20):
    env = gym.make('MountainCar-v0')
    test_rewards, test_lengths = [], []

    print(f"\nTesting trained agent ({n_episodes} episodes, ε=0)...")
    for episode in range(n_episodes):
        state, _ = env.reset()
        state    = discretize(state)
        total_reward = 0
        steps        = 0

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
    print(f"\n  Average : {avg:.2f} ± {np.std(test_rewards):.2f}")
    print(f"  Success : {success}/{n_episodes}  ({100*success/n_episodes:.0f}%)")
    print(f"  Result  : {'SOLVED' if avg >= SOLVED_AVG else 'Needs more training'}")
    env.close()


if __name__ == "__main__":
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/mountaincar_qlearning_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    q_table, rewards, lengths, epsilons, solve_ep = train()

    save_log(rewards, lengths, epsilons, results_dir)
    save_summary(rewards, lengths, solve_ep, results_dir)
    plot_results(rewards, lengths, results_dir)
    plot_trajectory(q_table, results_dir)
    test_agent(q_table)

    np.save(f'{results_dir}/q_table.npy', q_table)
    print(f"\nResults saved to: {results_dir}/")
