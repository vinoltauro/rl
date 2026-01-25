"""
CartPole Q-Learning Implementation
This program trains an agent to balance a pole on a cart using Q-learning.
"""

import gymnasium as gym  # The RL environment library (formerly called 'gym')
import numpy as np       # For numerical operations and arrays
import matplotlib.pyplot as plt  # For plotting learning curves

# ============================================================================
# STEP 1: CREATE THE ENVIRONMENT
# ============================================================================

# gym.make() creates a CartPole environment instance
# 'CartPole-v1' is the environment ID - a standard benchmark problem
env = gym.make('CartPole-v1')

# Let's examine what we're working with
print("=" * 60)
print("ENVIRONMENT INFORMATION")
print("=" * 60)
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"Number of actions: {env.action_space.n}")
print()

# Reset the environment to get an initial state
state, info = env.reset(seed=42)  # seed for reproducibility
print("Sample initial state:", state)
print("State components:")
print(f"  [0] Cart Position: {state[0]:.4f}")
print(f"  [1] Cart Velocity: {state[1]:.4f}")
print(f"  [2] Pole Angle: {state[2]:.4f} radians ({np.degrees(state[2]):.2f} degrees)")
print(f"  [3] Pole Angular Velocity: {state[3]:.4f}")
print("=" * 60)
print()

"""
EXPLANATION OF STATE:
Think of the state as 4 sensors giving real-time measurements:
- Cart Position: How far left/right the cart is from center
- Cart Velocity: How fast the cart is moving (positive = right)
- Pole Angle: How much the pole is tilted (0 = perfectly vertical)
- Pole Angular Velocity: How fast the pole is rotating

In GridWorld analogy: Instead of just (row, col), imagine you also know
your velocity, the treasure chest's wobble angle, and its wobble speed!
"""

# ============================================================================
# STEP 2: DISCRETIZATION - Converting Continuous to Discrete
# ============================================================================

print("DISCRETIZATION SETUP")
print("=" * 60)

# Number of bins (buckets) for each state dimension
# Think of bins as "buckets" that group similar continuous values together
# n_bins = [10, 10, 10, 10]  
n_bins = [10, 10, 20, 20]  
# This means: 10 bins for position, 10 for velocity, 10 for angle, 10 for angular velocity

"""
WHY DISCRETIZATION?
Q-learning needs a Q-table: Q(state, action)
- In GridWorld: state is discrete (like cell coordinates), so Q-table is straightforward
- In CartPole: state is continuous (infinite possibilities!), so we can't store infinite Q-values

SOLUTION: Divide the continuous range into discrete bins
Example: If cart position ranges from -4.8 to 4.8:
  Bin 0: -4.8 to -3.84
  Bin 1: -3.84 to -2.88
  Bin 2: -2.88 to -1.92
  ... and so on ...
  Bin 9: 3.84 to 4.8

All positions in Bin 0 are treated as "the same state"
"""

# Define the bounds for each state variable (from CartPole documentation)
state_bounds = [
    [-4.8, 4.8],        # Cart position (meters from center)
    [-4, 4],            # Cart velocity (m/s)
    [-0.418, 0.418],    # Pole angle (radians, ±24 degrees)
    [-4, 4]             # Pole angular velocity (rad/s)
]

print(f"Number of bins per dimension: {n_bins}")
print(f"State bounds: {state_bounds}")
print()

def create_bins(n_bins, state_bounds):
    """
    Creates bins (edges) for discretizing continuous state space.
    
    Args:
        n_bins: List of number of bins for each state dimension
        state_bounds: List of [min, max] bounds for each dimension
    
    Returns:
        List of arrays, each containing bin edges for one dimension
    
    Example: If n_bins[0]=3 and state_bounds[0]=[-4.8, 4.8]:
        bins[0] = [-4.8, -1.6, 1.6, 4.8]
        This creates 3 bins: [-4.8 to -1.6], [-1.6 to 1.6], [1.6 to 4.8]
    """
    bins = []
    for i in range(len(n_bins)):
        # np.linspace creates evenly spaced values between min and max
        bins.append(np.linspace(state_bounds[i][0], state_bounds[i][1], n_bins[i]))
    return bins

# Create the bins
bins = create_bins(n_bins, state_bounds)

print("Bins created for each dimension:")
for i, bin_edges in enumerate(bins):
    print(f"  Dimension {i}: {bin_edges}")
print()

def discretize_state(state, bins):
    """
    Converts a continuous state into a discrete state (tuple of bin indices).
    
    Args:
        state: Array of 4 continuous values
        bins: List of bin edges for each dimension
    
    Returns:
        Tuple of 4 integers representing which bin each value falls into
    
    Example:
        state = [1.5, -0.5, 0.1, 0.3]
        After discretization → (6, 4, 7, 6) 
        This means: position is in bin 6, velocity in bin 4, etc.
    
    GRIDWORLD ANALOGY:
    In GridWorld, your state is already discrete: (2, 3) means row 2, col 3
    Here, we're converting continuous coords like (1.47, 2.89) → (1, 3)
    """
    discrete_state = []
    for i in range(len(state)):
        # np.digitize finds which bin the value falls into
        # Subtract 1 because digitize returns 1-indexed, we want 0-indexed
        discrete_state.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(discrete_state)

# Test discretization
test_state = [1.5, -0.5, 0.1, 0.3]
discrete_test = discretize_state(test_state, bins)
print(f"Example discretization:")
print(f"  Continuous state: {test_state}")
print(f"  Discrete state: {discrete_test}")
print("=" * 60)
print()

# ============================================================================
# STEP 3: INITIALIZE Q-TABLE
# ============================================================================

print("Q-TABLE INITIALIZATION")
print("=" * 60)

"""
THE Q-TABLE: The Brain of Q-Learning

Q(state, action) = Expected total reward for taking 'action' in 'state'

GRIDWORLD ANALOGY:
If you have a 5×5 grid and 4 actions:
Q-table shape: (5, 5, 4)
Q[2][3][UP] = "How good is it to go UP when at position (2,3)?"

CARTPOLE:
We have 10×10×10×10 discrete states and 2 actions:
Q-table shape: (10, 10, 10, 10, 2)
Q[3][5][7][2][LEFT] = "How good is it to push LEFT when in discrete state (3,5,7,2)?"
"""

# Create Q-table with dimensions matching our discretized state space + actions
q_table_shape = n_bins + [env.action_space.n]  # [10, 10, 10, 10, 2]
q_table = np.zeros(q_table_shape)  # Initialize all Q-values to 0

print(f"Q-table shape: {q_table.shape}")
print(f"Total Q-values: {np.prod(q_table_shape):,}")
print(f"Memory size: ~{q_table.nbytes / 1024:.2f} KB")
print()

"""
WHY INITIALIZE TO ZERO?
The agent starts with no knowledge (optimistic initialization).
As it explores, it will update these values based on actual rewards received.
"""

# ============================================================================
# STEP 4: HYPERPARAMETERS - The Knobs We Turn
# ============================================================================

print("HYPERPARAMETERS")
print("=" * 60)

# Learning rate (α - alpha): How much to update Q-values after each step
# Range: 0 to 1
# Low (0.01): Slow learning, more stable
# High (0.9): Fast learning, more volatile
learning_rate = 0.1

"""
LEARNING RATE INTUITION:
Imagine you're adjusting your belief about "how good is this action?"
- α=0: Never learn from new experiences (stuck with initial beliefs)
- α=1: Completely replace old belief with new experience (no memory)
- α=0.1: Blend 10% new experience with 90% old belief (balanced)

Update formula: new_Q = old_Q + α × (error)
"""

# Discount factor (γ - gamma): How much to value future rewards
# Range: 0 to 1
# Low (0.1): Only care about immediate rewards (myopic)
# High (0.99): Care about long-term rewards (far-sighted)
discount_factor = 0.99

"""
DISCOUNT FACTOR INTUITION:
Would you prefer $10 now or $15 next year?
- γ=0: Only care about immediate reward (take $10 now)
- γ=1: Future rewards are equally valuable (prefer $15)
- γ=0.99: Future rewards are slightly less valuable

In CartPole: High γ makes the agent plan ahead to keep pole balanced longer
"""

# Exploration rate (ε - epsilon): Probability of taking random action
# Starts high (explore) → decays to low (exploit learned knowledge)
epsilon = 1.0          # Start: 100% exploration
epsilon_decay = 0.995  # Multiply epsilon by this after each episode
epsilon_min = 0.01     # Minimum: always keep 1% exploration

"""
EPSILON-GREEDY STRATEGY:
With probability ε: explore (random action)
With probability 1-ε: exploit (best known action)

Episode 1: ε=1.0 → 100% random (pure exploration)
Episode 100: ε≈0.6 → 60% random, 40% using learned policy
Episode 1000: ε≈0.01 → 1% random, 99% using learned policy

WHY DECAY?
Early: Don't know anything → explore to gather data
Late: Know a lot → exploit what we learned, explore occasionally
"""

# Training parameters
n_episodes = 10000  # Number of training episodes (games to play)
max_steps = 500     # Maximum steps per episode (CartPole-v1 limit)

print(f"Learning rate (α): {learning_rate}")
print(f"Discount factor (γ): {discount_factor}")
print(f"Initial epsilon (ε): {epsilon}")
print(f"Epsilon decay: {epsilon_decay}")
print(f"Minimum epsilon: {epsilon_min}")
print(f"Number of episodes: {n_episodes:,}")
print(f"Max steps per episode: {max_steps}")
print("=" * 60)
print()

# ============================================================================
# STEP 5: ACTION SELECTION - Epsilon-Greedy Strategy
# ============================================================================

def choose_action(state, epsilon):
    """
    Choose action using epsilon-greedy strategy.
    
    Args:
        state: Current discrete state (tuple of bin indices)
        epsilon: Current exploration rate
    
    Returns:
        action: 0 (push left) or 1 (push right)
    
    PROCESS:
    1. Generate random number between 0 and 1
    2. If random < epsilon: EXPLORE (random action)
    3. Else: EXPLOIT (action with highest Q-value)
    
    GRIDWORLD ANALOGY:
    You're at position (2,3) and can go UP/DOWN/LEFT/RIGHT
    - Explore: Close your eyes and pick a random direction
    - Exploit: Look at your map (Q-table) and go the direction marked "best"
    """
    if np.random.random() < epsilon:
        # EXPLORE: Random action
        return env.action_space.sample()  # Returns 0 or 1 randomly
    else:
        # EXPLOIT: Best action according to current Q-table
        return np.argmax(q_table[state])  # Returns action with highest Q-value

"""
WHY EPSILON-GREEDY?
Problem: If we only exploit, we might miss better strategies
Problem: If we only explore, we never use what we learned

Solution: Balance both!
- Early training: High epsilon → mostly explore
- Late training: Low epsilon → mostly exploit
- Always keep small epsilon → occasionally discover new things
"""

# ============================================================================
# STEP 6: Q-VALUE UPDATE - The Heart of Q-Learning
# ============================================================================

def update_q_table(state, action, reward, next_state, done):
    """
    Update Q-value using the Q-learning update rule (Bellman equation).
    
    Args:
        state: Current discrete state
        action: Action taken (0 or 1)
        reward: Reward received (usually 1.0 in CartPole)
        next_state: Next discrete state after taking action
        done: Boolean, True if episode ended
    
    Q-LEARNING FORMULA:
    Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
    
    Breaking it down:
    - Q(s,a): Current Q-value for state-action pair
    - r: Immediate reward
    - γ·max Q(s',a'): Discounted maximum future reward
    - [r + γ·max Q(s',a') - Q(s,a)]: TD error (prediction error)
    - α: How much to adjust based on this error
    
    GRIDWORLD ANALOGY:
    You're at (2,3), go RIGHT, get reward +10, end up at (2,4)
    Current belief: Q[(2,3), RIGHT] = 5
    
    What should it be?
    = immediate reward + discounted future value
    = 10 + 0.99 × max(Q[(2,4), all actions])
    = 10 + 0.99 × 15  (assuming best action from (2,4) has Q=15)
    = 24.85
    
    But Q was 5, so we were wrong!
    Error = 24.85 - 5 = 19.85
    New Q = 5 + 0.1 × 19.85 = 6.985
    
    We adjusted our belief upward based on this new experience!
    """
    
    # Get current Q-value for this state-action pair
    # state is a tuple like (3, 5, 7, 2), action is 0 or 1
    # state + (action,) creates (3, 5, 7, 2, 0) or (3, 5, 7, 2, 1)
    current_q = q_table[state + (action,)]
    
    if done:
        # Episode ended - no future rewards possible
        max_next_q = 0
    else:
        # Find maximum Q-value for next state (best action we could take from next state)
        max_next_q = np.max(q_table[next_state])
    
    # Calculate target Q-value
    target_q = reward + discount_factor * max_next_q
    
    # Calculate TD (Temporal Difference) error
    td_error = target_q - current_q
    
    # Update Q-value: blend old value with new target
    new_q = current_q + learning_rate * td_error
    
    # Store updated Q-value back in table
    q_table[state + (action,)] = new_q

"""
KEY INSIGHT: BOOTSTRAPPING
Q-learning is "bootstrapping" - it uses its own estimates to update itself!
- We don't wait for the episode to end
- We update after each step using our current estimate of future value
- Over time, these estimates become more accurate
"""

# ============================================================================
# STEP 7: TRAINING LOOP - Where the Magic Happens
# ============================================================================

print("STARTING TRAINING")
print("=" * 60)

# Track rewards for each episode (for plotting later)
episode_rewards = []

# Main training loop: play n_episodes games
for episode in range(n_episodes):
    # Reset environment to starting position
    # Returns initial state and info dict
    state, info = env.reset()
    
    # Convert continuous state to discrete bins
    state = discretize_state(state, bins)
    
    # Track total reward for this episode
    total_reward = 0
    
    # Episode loop: take steps until pole falls or max steps reached
    for step in range(max_steps):
        # 1. Choose action (explore or exploit)
        action = choose_action(state, epsilon)
        
        # 2. Take action in environment
        # Returns: next_state, reward, terminated, truncated, info
        #   - terminated: Episode ended naturally (pole fell)
        #   - truncated: Episode cut off by time limit
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # 3. Discretize next state
        next_state = discretize_state(next_state, bins)
        
        # 4. Check if episode is done
        done = terminated or truncated
        
        # 5. Update Q-table with this experience
        update_q_table(state, action, reward, next_state, done)
        
        # 6. Move to next state
        state = next_state
        total_reward += reward
        
        # 7. If episode ended, break out of step loop
        if done:
            break
    
    # After episode ends:
    
    # Decay epsilon (explore less over time)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # Store this episode's reward
    episode_rewards.append(total_reward)
    
    # Print progress every 100 episodes
    if (episode + 1) % 100 == 0:
        # Calculate average reward over last 100 episodes
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episode {episode + 1:5d}/{n_episodes} | "
              f"Avg Reward: {avg_reward:6.2f} | "
              f"Epsilon: {epsilon:.3f} | "
              f"Last Episode: {total_reward:3.0f} steps")

print("=" * 60)
print("TRAINING COMPLETED!")
print()

"""
WHAT HAPPENS DURING TRAINING?

Early Episodes (e.g., Episode 1-1000):
- High epsilon → mostly random actions
- Pole falls quickly (reward ≈ 10-30)
- Q-table starts to fill with values
- Agent learns basic associations: "this state-action led to quick failure"

Middle Episodes (e.g., Episode 1000-5000):
- Medium epsilon → mix of random and learned actions
- Pole stays up longer (reward ≈ 50-150)
- Q-table values become more accurate
- Agent learns better strategies

Late Episodes (e.g., Episode 5000-10000):
- Low epsilon → mostly using learned policy
- Pole stays up much longer (reward ≈ 200-500)
- Q-table converges to good values
- Agent has learned a solid policy

CartPole is "solved" when average reward ≥ 195 over 100 episodes!
"""

# ============================================================================
# STEP 8: VISUALIZE LEARNING PROGRESS
# ============================================================================

print("GENERATING PLOTS")
print("=" * 60)

# Create figure with two subplots side by side
plt.figure(figsize=(15, 5))

# LEFT PLOT: All episode rewards
plt.subplot(1, 2, 1)
plt.plot(episode_rewards, alpha=0.6, color='blue')
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Total Reward (Steps Balanced)', fontsize=12)
plt.title('Reward per Episode', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add horizontal line at "solved" threshold
plt.axhline(y=195, color='red', linestyle='--', label='Solved Threshold (195)')
plt.legend()

# RIGHT PLOT: Moving average (smoothed curve)
plt.subplot(1, 2, 2)
window = 100  # Average over 100 episodes
# np.convolve computes moving average
moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
plt.plot(moving_avg, color='green', linewidth=2)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Average Reward (100-episode window)', fontsize=12)
plt.title(f'Moving Average (window={window})', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add horizontal line at "solved" threshold
plt.axhline(y=195, color='red', linestyle='--', label='Solved Threshold (195)')
plt.legend()

plt.tight_layout()
plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
print("Plot saved as 'training_progress.png'")
plt.show()

"""
INTERPRETING THE PLOTS:

Left plot (Raw rewards):
- Shows actual reward for each episode
- Very noisy because of randomness
- Trend should go upward over time

Right plot (Moving average):
- Smooths out the noise
- Clearer view of learning progress
- Should show steady improvement

If training worked:
- Moving average should reach ~195+ 
- Left plot should show most episodes above 195 by the end
"""

# ============================================================================
# STEP 9: TEST THE TRAINED AGENT
# ============================================================================

print("\nTESTING TRAINED AGENT")
print("=" * 60)

def test_agent(n_test_episodes=10):
    """
    Test the trained agent without exploration (epsilon=0).
    
    This shows how well the agent learned by using only its
    learned policy (no random actions).
    """
    test_rewards = []
    
    for episode in range(n_test_episodes):
        state, info = env.reset()
        state = discretize_state(state, bins)
        total_reward = 0
        
        for step in range(max_steps):
            # Always choose best action (pure exploitation, no exploration)
            action = np.argmax(q_table[state])
            
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = discretize_state(next_state, bins)
            
            state = next_state
            total_reward += reward
            
            if terminated or truncated:
                break
        
        test_rewards.append(total_reward)
        print(f"  Test Episode {episode + 1:2d}: {total_reward:3.0f} steps")
    
    avg_test_reward = np.mean(test_rewards)
    print(f"\n  Average Test Reward: {avg_test_reward:.2f}")
    
    if avg_test_reward >= 195:
        print("  ✓ Agent has SOLVED CartPole! 🎉")
    else:
        print(f"  ✗ Agent needs more training (target: 195+)")
    
    return test_rewards

# Run tests
test_rewards = test_agent(n_test_episodes=10)

# ============================================================================
# STEP 10: VISUALIZE THE AGENT (OPTIONAL)
# ============================================================================

print("\n" + "=" * 60)
print("VISUALIZATION")
print("=" * 60)
print("To see the trained agent in action, uncomment the code below.")
print("It will open a window showing the cart and pole.")
print()

# Uncomment this section to watch your trained agent!

env_visual = gym.make('CartPole-v1', render_mode='human')
state, info = env_visual.reset()
state = discretize_state(state, bins)

print("Watch the agent balance the pole!")
print("Close the window when done.")

for step in range(500):
    action = np.argmax(q_table[state])
    next_state, reward, terminated, truncated, info = env_visual.step(action)
    next_state = discretize_state(next_state, bins)
    state = next_state
    
    if terminated or truncated:
        print(f"Episode ended after {step + 1} steps")
        break

env_visual.close()

# Clean up
env.close()

print("=" * 60)
print("PROGRAM FINISHED")
print("=" * 60)