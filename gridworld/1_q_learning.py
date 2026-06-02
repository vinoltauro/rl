import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import os

SEED = 42
np.random.seed(SEED)

# Grid
GRID_H, GRID_W = 6, 6
START = (0, 0)
GOAL  = (5, 5)
WALLS = {(1, 1), (1, 3), (2, 1), (3, 3), (3, 4), (4, 2)}

# Actions: 0=up, 1=right, 2=down, 3=left
N_ACTIONS     = 4
DELTAS        = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ACTION_ARROWS = ['↑', '→', '↓', '←']

# Hyperparameters
LR         = 0.1
GAMMA      = 0.99
EPS_START  = 1.0
EPS_MIN    = 0.01
EPS_DECAY  = 0.995
N_EPISODES = 3000
MAX_STEPS  = 200
REWARD_GOAL = 10.0
REWARD_STEP = -0.1
REWARD_WALL = -1.0


def env_step(state, action):
    r, c = state
    dr, dc = DELTAS[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < GRID_H and 0 <= nc < GRID_W) or (nr, nc) in WALLS:
        return state, REWARD_WALL, False
    next_state = (nr, nc)
    if next_state == GOAL:
        return next_state, REWARD_GOAL, True
    return next_state, REWARD_STEP, False


def choose_action(q_table, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(N_ACTIONS)
    return int(np.argmax(q_table[state[0], state[1]]))


def train():
    q_table = np.zeros((GRID_H, GRID_W, N_ACTIONS))
    epsilon = EPS_START
    episode_rewards = []
    episode_lengths = []

    print("GridWorld Q-Learning")
    print(f"Grid: {GRID_H}x{GRID_W} | Walls: {len(WALLS)} | Start: {START} | Goal: {GOAL}")
    print("=" * 50)

    for episode in range(N_EPISODES):
        state = START
        total_reward = 0

        for step in range(MAX_STEPS):
            action = choose_action(q_table, state, epsilon)
            next_state, reward, done = env_step(state, action)

            current_q  = q_table[state[0], state[1], action]
            max_next_q = 0.0 if done else np.max(q_table[next_state[0], next_state[1]])
            q_table[state[0], state[1], action] += LR * (reward + GAMMA * max_next_q - current_q)

            state = next_state
            total_reward += reward
            if done:
                break

        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)
        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)

        if (episode + 1) % 300 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1:4d} | Avg reward (100): {avg:6.2f} | Epsilon: {epsilon:.3f}")

    return q_table, episode_rewards, episode_lengths


def plot_results(q_table, episode_rewards, episode_lengths, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Raw rewards
    axes[0, 0].plot(episode_rewards, alpha=0.5, color='blue', linewidth=0.5)
    window = 100
    if len(episode_rewards) >= window:
        ma = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        axes[0, 0].plot(range(window - 1, len(episode_rewards)), ma, color='red', linewidth=2, label='100-ep avg')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Episode lengths
    axes[0, 1].plot(episode_lengths, alpha=0.5, color='orange', linewidth=0.5)
    if len(episode_lengths) >= window:
        ma_len = np.convolve(episode_lengths, np.ones(window) / window, mode='valid')
        axes[0, 1].plot(range(window - 1, len(episode_lengths)), ma_len, color='darkred', linewidth=2, label='100-ep avg')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Steps to Goal')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Value function heatmap
    value_map = np.max(q_table, axis=2).copy()
    for r, c in WALLS:
        value_map[r, c] = np.nan
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color='black')
    im = axes[1, 0].imshow(value_map, cmap=cmap, interpolation='nearest')
    plt.colorbar(im, ax=axes[1, 0])
    axes[1, 0].set_title('Value Function  max_a Q(s,a)')
    axes[1, 0].text(START[1], START[0], 'S', ha='center', va='center', fontsize=14, fontweight='bold', color='blue')
    axes[1, 0].text(GOAL[1], GOAL[0], 'G', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    # Greedy policy arrows
    ax = axes[1, 1]
    ax.set_xlim(-0.5, GRID_W - 0.5)
    ax.set_ylim(GRID_H - 0.5, -0.5)
    ax.set_title('Greedy Policy')
    ax.set_xticks(range(GRID_W))
    ax.set_yticks(range(GRID_H))
    ax.grid(True, alpha=0.3)
    for r in range(GRID_H):
        for c in range(GRID_W):
            if (r, c) in WALLS:
                ax.add_patch(patches.Rectangle((c - 0.5, r - 0.5), 1, 1, color='black'))
            elif (r, c) == GOAL:
                ax.text(c, r, 'G', ha='center', va='center', fontsize=14, fontweight='bold', color='green')
            else:
                best = int(np.argmax(q_table[r, c]))
                color = 'blue' if (r, c) == START else 'black'
                ax.text(c, r, ACTION_ARROWS[best], ha='center', va='center', fontsize=14, color=color)

    plt.suptitle('GridWorld Q-Learning', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/training_results.png")


def test_agent(q_table, n_episodes=10):
    print(f"\nTesting trained agent ({n_episodes} episodes, epsilon=0)...")
    rewards, lengths = [], []

    for episode in range(n_episodes):
        state = START
        total_reward = 0

        for step in range(MAX_STEPS):
            action = int(np.argmax(q_table[state[0], state[1]]))
            state, reward, done = env_step(state, action)
            total_reward += reward
            if done:
                break

        rewards.append(total_reward)
        lengths.append(step + 1)
        status = "✓" if state == GOAL else "✗"
        print(f"  {status} Test {episode+1:2d}: {total_reward:6.2f} reward  |  {step+1:3d} steps")

    print(f"\n  Avg reward : {np.mean(rewards):.2f}")
    print(f"  Avg steps  : {np.mean(lengths):.1f}")
    print(f"  Reached goal: {sum(1 for r in rewards if r > 0)}/{n_episodes}")


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/gridworld_qlearning_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    q_table, rewards, lengths = train()
    plot_results(q_table, rewards, lengths, results_dir)
    test_agent(q_table)

    np.save(f'{results_dir}/q_table.npy', q_table)
    print(f"\nResults saved to: {results_dir}/")
