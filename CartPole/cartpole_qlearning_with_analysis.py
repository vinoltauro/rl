import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

env = gym.make('CartPole-v1')

n_bins = [8, 10, 20, 20]
state_bounds = [
    [-2.4, 2.4],
    [-3, 3],
    [-0.3, 0.3],
    [-3, 3]
]

def create_bins(n_bins, state_bounds):
    bins = []
    for i in range(len(n_bins)):
        bins.append(np.linspace(state_bounds[i][0], state_bounds[i][1], n_bins[i]))
    return bins

def discretize_state(state, bins):
    discrete_state = []
    for i in range(len(state)):
        discrete_state.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(discrete_state)

bins = create_bins(n_bins, state_bounds)

q_table_shape = n_bins + [env.action_space.n]
q_table = np.zeros(q_table_shape)

learning_rate = 0.15
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.01
n_episodes = 15000
max_steps = 500
solved_threshold = 195
stop_when_solved = True

def choose_action(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

def update_q_table(state, action, reward, next_state, done):
    current_q = q_table[state + (action,)]
    max_next_q = 0 if done else np.max(q_table[next_state])
    new_q = current_q + learning_rate * (reward + discount_factor * max_next_q - current_q)
    q_table[state + (action,)] = new_q

episode_rewards = []
epsilon_history = []
episode_lengths = []

print("="*60)
print("CARTPOLE Q-LEARNING WITH COMPREHENSIVE ANALYSIS")
print("="*60)
print(f"Configuration:")
print(f"  Bins: {n_bins}")
print(f"  Learning rate: {learning_rate}")
print(f"  Discount factor: {discount_factor}")
print(f"  Epsilon decay: {epsilon_decay}")
print(f"  Max episodes: {n_episodes}")
print("="*60)

start_time = time.time()

for episode in range(n_episodes):
    state, info = env.reset()
    state = discretize_state(state, bins)
    total_reward = 0
    steps = 0
    
    for step in range(max_steps):
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = discretize_state(next_state, bins)
        done = terminated or truncated
        
        update_q_table(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)
    epsilon_history.append(epsilon)
    episode_lengths.append(steps)
    
    if stop_when_solved and len(episode_rewards) >= 100:
        recent_avg = np.mean(episode_rewards[-100:])
        if recent_avg >= solved_threshold:
            print(f"\n🎉 CartPole SOLVED at episode {episode + 1}!")
            print(f"Average reward: {recent_avg:.2f}")
            print(f"Training time: {time.time() - start_time:.2f} seconds")
            break
    
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {episode + 1:5d} | Avg: {avg_reward:6.2f} | Epsilon: {epsilon:.3f} | Last: {total_reward:3.0f}")

training_time = time.time() - start_time
print("="*60)
print(f"TRAINING COMPLETED in {training_time:.2f} seconds")
print("="*60)

print("\nGenerating comprehensive analysis plots...")

fig = plt.figure(figsize=(20, 12))

plt.subplot(3, 3, 1)
plt.plot(episode_rewards, alpha=0.6, color='blue', linewidth=0.5)
plt.xlabel('Episode', fontsize=11)
plt.ylabel('Total Reward (Steps)', fontsize=11)
plt.title('Raw Rewards per Episode', fontsize=12, fontweight='bold')
plt.axhline(y=195, color='red', linestyle='--', linewidth=2, label='Solved (195)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 2)
window = 100
moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
plt.plot(moving_avg, color='green', linewidth=2)
plt.xlabel('Episode', fontsize=11)
plt.ylabel('Average Reward', fontsize=11)
plt.title(f'Moving Average (window={window})', fontsize=12, fontweight='bold')
plt.axhline(y=195, color='red', linestyle='--', linewidth=2, label='Solved (195)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 3)
plt.plot(epsilon_history, color='orange', linewidth=2)
plt.xlabel('Episode', fontsize=11)
plt.ylabel('Epsilon', fontsize=11)
plt.title('Exploration Rate (Epsilon) Decay', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 4)
window_sizes = [10, 50, 100, 200]
for window in window_sizes:
    if len(episode_rewards) >= window:
        ma = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(ma, linewidth=1.5, label=f'Window {window}', alpha=0.8)
plt.xlabel('Episode', fontsize=11)
plt.ylabel('Average Reward', fontsize=11)
plt.title('Multiple Moving Averages', fontsize=12, fontweight='bold')
plt.axhline(y=195, color='red', linestyle='--', linewidth=2, alpha=0.5)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 5)
bins_hist = np.linspace(0, 500, 50)
plt.hist(episode_rewards, bins=bins_hist, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(x=195, color='red', linestyle='--', linewidth=2, label='Solved (195)')
plt.xlabel('Reward', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Distribution of Episode Rewards', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(3, 3, 6)
segment_size = max(1, len(episode_rewards) // 5)
segments = [episode_rewards[i:i+segment_size] for i in range(0, len(episode_rewards), segment_size)]
segment_labels = [f'Ep {i*segment_size}-{min((i+1)*segment_size, len(episode_rewards))}' for i in range(len(segments))]
plt.boxplot(segments, labels=range(1, len(segments)+1))
plt.axhline(y=195, color='red', linestyle='--', linewidth=2, label='Solved (195)')
plt.xlabel('Training Phase', fontsize=11)
plt.ylabel('Reward', fontsize=11)
plt.title('Reward Distribution Over Training Phases', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(3, 3, 7)
success_rate = []
window = 100
for i in range(window, len(episode_rewards) + 1):
    success_rate.append(np.mean(np.array(episode_rewards[i-window:i]) >= 195) * 100)
plt.plot(range(window, len(episode_rewards) + 1), success_rate, color='purple', linewidth=2)
plt.xlabel('Episode', fontsize=11)
plt.ylabel('Success Rate (%)', fontsize=11)
plt.title(f'Success Rate (% episodes ≥195, window={window})', fontsize=12, fontweight='bold')
plt.axhline(y=50, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='50%')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 8)
max_rewards = []
window = 100
for i in range(window, len(episode_rewards) + 1):
    max_rewards.append(np.max(episode_rewards[i-window:i]))
plt.plot(range(window, len(episode_rewards) + 1), max_rewards, color='darkgreen', linewidth=2)
plt.xlabel('Episode', fontsize=11)
plt.ylabel('Max Reward', fontsize=11)
plt.title(f'Best Performance (max reward in {window}-episode windows)', fontsize=12, fontweight='bold')
plt.axhline(y=500, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Max Possible (500)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 3, 9)
cumulative_reward = np.cumsum(episode_rewards)
plt.plot(cumulative_reward, color='teal', linewidth=2)
plt.xlabel('Episode', fontsize=11)
plt.ylabel('Cumulative Reward', fontsize=11)
plt.title('Cumulative Reward Over Training', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.suptitle('CartPole Q-Learning: Comprehensive Training Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('comprehensive_training_analysis.png', dpi=200, bbox_inches='tight')
print("Saved: comprehensive_training_analysis.png")
plt.show()

fig2, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].plot(episode_rewards, alpha=0.6, color='blue', linewidth=0.5)
moving_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
axes[0, 0].plot(range(99, 99+len(moving_avg)), moving_avg, color='red', linewidth=2, label='100-ep MA')
axes[0, 0].axhline(y=195, color='green', linestyle='--', linewidth=2, label='Solved')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Reward')
axes[0, 0].set_title('Training Progress with Moving Average')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

q_values_left = []
q_values_right = []
for state_idx in range(min(1000, np.prod(n_bins))):
    indices = np.unravel_index(state_idx, n_bins)
    q_values_left.append(q_table[indices + (0,)])
    q_values_right.append(q_table[indices + (1,)])
axes[0, 1].hist(q_values_left, bins=50, alpha=0.6, label='Push Left', color='blue')
axes[0, 1].hist(q_values_right, bins=50, alpha=0.6, label='Push Right', color='red')
axes[0, 1].set_xlabel('Q-Value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Q-Values by Action')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

early = episode_rewards[:len(episode_rewards)//3]
middle = episode_rewards[len(episode_rewards)//3:2*len(episode_rewards)//3]
late = episode_rewards[2*len(episode_rewards)//3:]
axes[1, 0].violinplot([early, middle, late], positions=[1, 2, 3], showmeans=True)
axes[1, 0].axhline(y=195, color='red', linestyle='--', linewidth=2, label='Solved')
axes[1, 0].set_xticks([1, 2, 3])
axes[1, 0].set_xticklabels(['Early', 'Middle', 'Late'])
axes[1, 0].set_ylabel('Reward')
axes[1, 0].set_title('Performance Distribution Across Training Phases')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

metrics_text = f"""
TRAINING SUMMARY
================

Configuration:
• Bins: {n_bins}
• Learning Rate: {learning_rate}
• Discount Factor: {discount_factor}
• Epsilon Decay: {epsilon_decay}

Results:
• Total Episodes: {len(episode_rewards)}
• Training Time: {training_time:.2f}s
• Final Epsilon: {epsilon_history[-1]:.4f}

Performance:
• First 100 avg: {np.mean(episode_rewards[:100]):.2f}
• Last 100 avg: {np.mean(episode_rewards[-100:]):.2f}
• Best 100 avg: {np.max([np.mean(episode_rewards[i:i+100]) for i in range(len(episode_rewards)-100)]):.2f}
• Max episode: {np.max(episode_rewards):.0f}
• Min episode: {np.min(episode_rewards):.0f}

Success Metrics:
• Episodes ≥195: {np.sum(np.array(episode_rewards) >= 195)} ({100*np.sum(np.array(episode_rewards) >= 195)/len(episode_rewards):.1f}%)
• Episodes ≥400: {np.sum(np.array(episode_rewards) >= 400)} ({100*np.sum(np.array(episode_rewards) >= 400)/len(episode_rewards):.1f}%)
"""

axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                fontsize=10, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 1].axis('off')
axes[1, 1].set_title('Training Statistics')

plt.suptitle('CartPole Q-Learning: Detailed Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('detailed_analysis.png', dpi=200, bbox_inches='tight')
print("Saved: detailed_analysis.png")
plt.show()

print("\nTesting trained agent...")
test_rewards = []

for episode in range(10):
    state, info = env.reset()
    state = discretize_state(state, bins)
    total_reward = 0
    
    for step in range(max_steps):
        action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = discretize_state(next_state, bins)
        state = next_state
        total_reward += reward
        
        if terminated or truncated:
            break
    
    test_rewards.append(total_reward)
    print(f"  Test Episode {episode + 1:2d}: {total_reward:3.0f} steps")

avg_test = np.mean(test_rewards)
std_test = np.std(test_rewards)
print(f"\nTest Results:")
print(f"  Average: {avg_test:.2f} ± {std_test:.2f}")
print(f"  Min: {np.min(test_rewards):.0f}")
print(f"  Max: {np.max(test_rewards):.0f}")

if avg_test >= 195:
    print(f"  ✓ Agent SOLVED CartPole!")
else:
    print(f"  ✗ Agent needs improvement (target: 195+)")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"Generated files:")
print(f"  • comprehensive_training_analysis.png")
print(f"  • detailed_analysis.png")

env.close()