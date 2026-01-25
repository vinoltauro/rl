import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')

n_bins = [40, 40]
state_bounds = [[-1.2, 0.6], [-0.07, 0.07]]

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
q_table = np.zeros(n_bins + [env.action_space.n])

learning_rate = 0.3
discount_factor = 0.995
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.001
n_episodes = 25000
max_steps = 200

def choose_action(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

def update_q_table(state, action, reward, next_state, done):
    current_q = q_table[state + (action,)]
    max_next_q = 0 if done and reward > -200 else np.max(q_table[next_state])
    new_q = current_q + learning_rate * (reward + discount_factor * max_next_q - current_q)
    q_table[state + (action,)] = new_q

episode_rewards = []
epsilon_history = []
best_avg_reward = -200
no_improvement_count = 0

print("="*60)
print("MOUNTAINCAR Q-LEARNING - FINAL VERSION")
print("="*60)

for episode in range(n_episodes):
    state, info = env.reset()
    state = discretize_state(state, bins)
    total_reward = 0
    
    for step in range(max_steps):
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = discretize_state(next_state, bins)
        done = terminated or truncated
        
        update_q_table(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)
    epsilon_history.append(epsilon)
    
    if len(episode_rewards) >= 100:
        avg_reward = np.mean(episode_rewards[-100:])
        
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if (episode + 1) % 500 == 0:
            success_rate = np.sum([1 for r in episode_rewards[-100:] if r > -200])
            print(f"Ep {episode+1:5d} | Avg: {avg_reward:7.2f} | Best: {best_avg_reward:7.2f} | Success: {success_rate}/100 | ε: {epsilon:.4f}")
        
        if avg_reward >= -110 and episode > 8000:
            print(f"\n🎉 SOLVED at episode {episode + 1}!")
            print(f"Average reward: {avg_reward:.2f}")
            break
        
        if no_improvement_count >= 2000 and episode > 10000:
            print(f"\n⚠ Stopping: No improvement for 2000 episodes")
            print(f"Best average: {best_avg_reward:.2f}")
            break

print("="*60)
print(f"TRAINING COMPLETED - Episodes: {len(episode_rewards)}")
print(f"Best 100-episode average: {best_avg_reward:.2f}")
print("="*60)

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(episode_rewards, alpha=0.6, color='blue', linewidth=0.5)
plt.axhline(y=-110, color='red', linestyle='--', linewidth=2, label='Solved (-110)')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward per Episode')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
window = 100
moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
plt.plot(moving_avg, color='green', linewidth=2)
plt.axhline(y=-110, color='red', linestyle='--', linewidth=2, label='Solved (-110)')
best_idx = np.argmax(moving_avg)
plt.plot(best_idx, moving_avg[best_idx], 'r*', markersize=15, label=f'Best: {moving_avg[best_idx]:.1f}')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Moving Average (100-ep window)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
episode_lengths = [abs(r) for r in episode_rewards]
plt.plot(episode_lengths, alpha=0.6, color='red', linewidth=0.5)
plt.axhline(y=110, color='green', linestyle='--', linewidth=2, label='Target (<110)')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
moving_avg_steps = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
plt.plot(moving_avg_steps, color='orange', linewidth=2)
plt.axhline(y=110, color='green', linestyle='--', linewidth=2, label='Target (<110)')
best_steps_idx = np.argmin(moving_avg_steps)
plt.plot(best_steps_idx, moving_avg_steps[best_steps_idx], 'g*', markersize=15, 
         label=f'Best: {moving_avg_steps[best_steps_idx]:.1f}')
plt.xlabel('Episode')
plt.ylabel('Average Steps')
plt.title('Moving Average Steps')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
plt.plot(epsilon_history, color='purple', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Exploration Rate')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
plt.hist(episode_rewards, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(x=-110, color='red', linestyle='--', linewidth=2, label='Solved (-110)')
plt.axvline(x=np.mean(episode_rewards), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(episode_rewards):.1f}')
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.title('Reward Distribution')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.suptitle('MountainCar Q-Learning: Final Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('mountaincar_final_results.png', dpi=150)
plt.show()

print("\nTesting trained agent (20 episodes)...")
test_rewards = []

for episode in range(20):
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
    status = "✓" if total_reward > -200 else "✗"
    print(f"{status} Test {episode+1:2d}: {total_reward:6.2f} ({abs(int(total_reward))} steps)")

avg_test = np.mean(test_rewards)
success_rate = np.sum([1 for r in test_rewards if r > -200])

print(f"\n{'='*60}")
print(f"Test Average: {avg_test:.2f} steps")
print(f"Success Rate: {success_rate}/20 ({100*success_rate/20:.0f}%)")

if avg_test >= -110:
    print("🎉 SOLVED! Agent consistently reaches goal efficiently!")
elif avg_test >= -130:
    print("⚠ Close! Good performance, nearly solved")
else:
    print("✗ More training needed")
print(f"{'='*60}")

env.close()