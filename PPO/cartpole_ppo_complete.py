import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import os
from datetime import datetime

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCritic, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value
    
    def get_action_probs(self, state):
        action_probs, _ = self.forward(state)
        return action_probs

class PPOAgent:
    def __init__(self, state_size, action_size, hidden_size=128):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.lam = 0.95
        self.epsilon = 0.2
        self.lr = 3e-4
        self.epochs = 10
        self.batch_size = 64
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, state_value = self.policy(state)
        
        if deterministic:
            action = action_probs.argmax().item()
            return action, 0.0, state_value.item()
        
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), state_value.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value):
        advantages = []
        gae = 0
        
        values = self.values + [next_value]
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return advantages, returns
    
    def update(self):
        if len(self.states) == 0:
            return 0, 0, 0, 0
        
        last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, next_value = self.policy(last_state)
            next_value = next_value.item()
        
        advantages, returns = self.compute_gae(next_value)
        
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clip_fraction = 0
        
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        num_batches = 0
        
        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                action_probs, state_values = self.policy(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.MSELoss()(state_values.squeeze(), batch_returns)
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.epsilon).float().mean()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_clip_fraction += clip_fraction.item()
                num_batches += 1
        
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        avg_entropy = total_entropy / num_batches
        avg_clip_fraction = total_clip_fraction / num_batches
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        return avg_policy_loss, avg_value_loss, avg_entropy, avg_clip_fraction

def train_ppo(n_episodes=500, update_freq=20, render_test=False):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = PPOAgent(state_size, action_size)
    
    episode_rewards = []
    policy_losses = []
    value_losses = []
    entropies = []
    clip_fractions = []
    episode_lengths = []
    
    print("="*70)
    print("TRAINING CARTPOLE WITH PPO (PROXIMAL POLICY OPTIMIZATION)")
    print("="*70)
    print(f"Device: {agent.device}")
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Hidden size: 128, Learning rate: {agent.lr}")
    print(f"Update frequency: every {update_freq} episodes")
    print("="*70)
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(500):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, log_prob, value, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if (episode + 1) % update_freq == 0:
            policy_loss, value_loss, entropy, clip_frac = agent.update()
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropies.append(entropy)
            clip_fractions.append(clip_frac)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            recent_avg_length = np.mean(episode_lengths[-10:])
            print(f"Ep {episode+1:4d} | R: {total_reward:3.0f} | Avg(100): {avg_reward:6.2f} | Steps: {recent_avg_length:5.1f}")
        
        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 195:
            print(f"\n{'='*70}")
            print(f"🎉 SOLVED at episode {episode + 1}!")
            print(f"Average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
            print(f"{'='*70}")
            break
    
    env.close()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"cartpole_ppo_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    create_comprehensive_plots(episode_rewards, episode_lengths, policy_losses, 
                               value_losses, entropies, clip_fractions, 
                               results_dir)
    
    test_results = test_agent(agent, render=render_test, save_dir=results_dir)
    
    save_results(episode_rewards, episode_lengths, policy_losses, value_losses, 
                entropies, clip_fractions, test_results, results_dir)
    
    torch.save(agent.policy.state_dict(), f'{results_dir}/ppo_model.pth')
    print(f"✓ Saved model: {results_dir}/ppo_model.pth")
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_dir}/")
    print(f"To watch trained agent: python watch_cartpole_agent.py {results_dir}/ppo_model.pth")
    print(f"{'='*70}")
    
    return agent, results_dir

def create_comprehensive_plots(episode_rewards, episode_lengths, policy_losses, 
                               value_losses, entropies, clip_fractions, save_dir):
    
    fig = plt.figure(figsize=(20, 12))
    
    plt.subplot(3, 4, 1)
    plt.plot(episode_rewards, alpha=0.4, color='blue', linewidth=0.5)
    plt.axhline(y=195, color='red', linestyle='--', linewidth=2, label='Solved (195)')
    plt.axhline(y=500, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Max (500)')
    plt.xlabel('Episode', fontsize=10)
    plt.ylabel('Reward', fontsize=10)
    plt.title('Episode Rewards', fontsize=12, fontweight='bold')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 2)
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_rewards)), moving_avg, color='green', linewidth=2)
        plt.axhline(y=195, color='red', linestyle='--', linewidth=2, label='Solved (195)')
        plt.fill_between(range(window-1, len(episode_rewards)), moving_avg-10, moving_avg+10, 
                        alpha=0.2, color='green')
    plt.xlabel('Episode', fontsize=10)
    plt.ylabel('Average Reward', fontsize=10)
    plt.title('Moving Average (100 episodes)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 3)
    plt.plot(episode_lengths, alpha=0.4, color='orange', linewidth=0.5)
    plt.xlabel('Episode', fontsize=10)
    plt.ylabel('Steps', fontsize=10)
    plt.title('Episode Lengths', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 4)
    if len(episode_lengths) >= 50:
        length_ma = np.convolve(episode_lengths, np.ones(50)/50, mode='valid')
        plt.plot(range(49, len(episode_lengths)), length_ma, color='darkorange', linewidth=2)
    plt.xlabel('Episode', fontsize=10)
    plt.ylabel('Average Steps', fontsize=10)
    plt.title('Moving Avg Steps (50 episodes)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 5)
    if policy_losses:
        plt.plot(policy_losses, color='red', linewidth=2, marker='o', markersize=3)
        plt.xlabel('Update', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.title('Policy Loss', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 6)
    if value_losses:
        plt.plot(value_losses, color='purple', linewidth=2, marker='o', markersize=3)
        plt.xlabel('Update', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.title('Value Loss', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 7)
    if entropies:
        plt.plot(entropies, color='green', linewidth=2, marker='o', markersize=3)
        plt.xlabel('Update', fontsize=10)
        plt.ylabel('Entropy', fontsize=10)
        plt.title('Policy Entropy', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 8)
    if clip_fractions:
        plt.plot(clip_fractions, color='brown', linewidth=2, marker='o', markersize=3)
        plt.axhline(y=0.2, color='red', linestyle='--', alpha=0.5)
        plt.xlabel('Update', fontsize=10)
        plt.ylabel('Clip Fraction', fontsize=10)
        plt.title('PPO Clipping Rate', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 9)
    plt.hist(episode_rewards, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=195, color='red', linestyle='--', linewidth=2, label='Solved (195)')
    plt.axvline(x=np.mean(episode_rewards), color='green', linestyle='-', linewidth=2, 
               label=f'Mean: {np.mean(episode_rewards):.1f}')
    plt.xlabel('Reward', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.title('Reward Distribution', fontsize=12, fontweight='bold')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(3, 4, 10)
    plt.hist(episode_lengths, bins=40, color='lightcoral', edgecolor='black', alpha=0.7)
    plt.axvline(x=np.mean(episode_lengths), color='darkred', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(episode_lengths):.1f}')
    plt.xlabel('Steps', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.title('Length Distribution', fontsize=12, fontweight='bold')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(3, 4, 11)
    if len(episode_rewards) >= 100:
        rolling_mean = []
        rolling_std = []
        for i in range(99, len(episode_rewards)):
            window_data = episode_rewards[i-99:i+1]
            rolling_mean.append(np.mean(window_data))
            rolling_std.append(np.std(window_data))
        
        x = range(99, len(episode_rewards))
        plt.plot(x, rolling_mean, color='blue', linewidth=2, label='Mean')
        plt.fill_between(x, 
                        np.array(rolling_mean) - np.array(rolling_std),
                        np.array(rolling_mean) + np.array(rolling_std),
                        alpha=0.3, color='blue', label='±1 Std')
        plt.axhline(y=195, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Episode', fontsize=10)
    plt.ylabel('Reward', fontsize=10)
    plt.title('Rolling Statistics (100-ep)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 4, 12)
    final_policy = f"{policy_losses[-1]:.4f}" if policy_losses else "N/A"
    final_value = f"{value_losses[-1]:.4f}" if value_losses else "N/A"
    final_entropy = f"{entropies[-1]:.4f}" if entropies else "N/A"
    
    stats_text = f"""
TRAINING SUMMARY
{'='*30}

Episodes: {len(episode_rewards)}
Updates: {len(policy_losses)}

Performance:
• First 10 avg: {np.mean(episode_rewards[:10]):.1f}
• Last 100 avg: {np.mean(episode_rewards[-100:]):.1f}
• Max reward: {np.max(episode_rewards):.0f}
• Mean reward: {np.mean(episode_rewards):.1f}

Efficiency:
• Mean steps: {np.mean(episode_lengths):.1f}
• Solved: {'Yes ✓' if np.mean(episode_rewards[-100:]) >= 195 else 'No ✗'}

Losses:
• Final policy: {final_policy}
• Final value: {final_value}
• Final entropy: {final_entropy}
    """
    
    plt.text(0.05, 0.5, stats_text, transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.axis('off')
    plt.title('Statistics', fontsize=12, fontweight='bold')
    
    plt.suptitle('CartPole PPO: Comprehensive Training Analysis', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plt.savefig(f'{save_dir}/training_analysis.png', dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {save_dir}/training_analysis.png")
    plt.close()

def test_agent(agent, n_episodes=20, render=False, save_dir=None):
    if render:
        env = gym.make('CartPole-v1', render_mode='human')
    else:
        env = gym.make('CartPole-v1')
    
    test_rewards = []
    test_lengths = []
    
    print(f"\n{'='*70}")
    print("TESTING TRAINED AGENT")
    print(f"{'='*70}")
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(500):
            action, _, _ = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        test_rewards.append(total_reward)
        test_lengths.append(steps)
        status = "✓" if total_reward >= 195 else "✗"
        print(f"{status} Test {episode+1:2d}: {total_reward:3.0f} steps")
    
    avg_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    success_rate = 100 * np.sum(np.array(test_rewards) >= 195) / n_episodes
    
    print(f"\n{'='*70}")
    print(f"Test Average: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Success Rate: {np.sum(np.array(test_rewards) >= 195)}/{n_episodes} ({success_rate:.0f}%)")
    print(f"Min: {np.min(test_rewards):.0f}, Max: {np.max(test_rewards):.0f}")
    
    if avg_reward >= 195:
        print("🎉 Agent SOLVED CartPole!")
    elif avg_reward >= 180:
        print("⚠ Very close to solving!")
    else:
        print("✗ Needs more training")
    print(f"{'='*70}")
    
    env.close()
    
    if save_dir:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].bar(range(1, n_episodes+1), test_rewards, 
                   color=['green' if r >= 195 else 'orange' for r in test_rewards],
                   alpha=0.7, edgecolor='black')
        axes[0].axhline(y=195, color='red', linestyle='--', linewidth=2, label='Target (195)')
        axes[0].axhline(y=avg_reward, color='blue', linestyle='-', linewidth=2, 
                       label=f'Mean: {avg_reward:.1f}')
        axes[0].set_xlabel('Test Episode', fontsize=12)
        axes[0].set_ylabel('Reward', fontsize=12)
        axes[0].set_title('Test Performance', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        axes[1].hist(test_rewards, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        axes[1].axvline(x=195, color='red', linestyle='--', linewidth=2, label='Target (195)')
        axes[1].axvline(x=avg_reward, color='blue', linestyle='-', linewidth=2,
                       label=f'Mean: {avg_reward:.1f}')
        axes[1].set_xlabel('Reward', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Test Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/test_results.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_dir}/test_results.png")
        plt.close()
    
    return {
        'rewards': test_rewards,
        'lengths': test_lengths,
        'mean': avg_reward,
        'std': std_reward,
        'success_rate': success_rate
    }

def save_results(episode_rewards, episode_lengths, policy_losses, value_losses,
                entropies, clip_fractions, test_results, save_dir):
    
    with open(f'{save_dir}/results_summary.txt', 'w') as f:
        f.write("CARTPOLE PPO TRAINING RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("TRAINING PERFORMANCE\n")
        f.write("-"*70 + "\n")
        f.write(f"Total Episodes: {len(episode_rewards)}\n")
        f.write(f"Total Updates: {len(policy_losses)}\n")
        f.write(f"Mean Reward: {np.mean(episode_rewards):.2f}\n")
        f.write(f"Std Reward: {np.std(episode_rewards):.2f}\n")
        f.write(f"Last 100 Avg: {np.mean(episode_rewards[-100:]):.2f}\n")
        f.write(f"Max Reward: {np.max(episode_rewards):.0f}\n")
        f.write(f"Mean Episode Length: {np.mean(episode_lengths):.2f}\n\n")
        
        f.write("TEST PERFORMANCE\n")
        f.write("-"*70 + "\n")
        f.write(f"Test Mean: {test_results['mean']:.2f}\n")
        f.write(f"Test Std: {test_results['std']:.2f}\n")
        f.write(f"Success Rate: {test_results['success_rate']:.1f}%\n")
        f.write(f"Min: {np.min(test_results['rewards']):.0f}\n")
        f.write(f"Max: {np.max(test_results['rewards']):.0f}\n\n")
        
        f.write("FINAL LOSSES\n")
        f.write("-"*70 + "\n")
        if policy_losses:
            f.write(f"Policy Loss: {policy_losses[-1]:.6f}\n")
            f.write(f"Value Loss: {value_losses[-1]:.6f}\n")
            f.write(f"Entropy: {entropies[-1]:.6f}\n")
            f.write(f"Clip Fraction: {clip_fractions[-1]:.6f}\n")
    
    print(f"✓ Saved: {save_dir}/results_summary.txt")
    
    np.savez(f'{save_dir}/training_data.npz',
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            policy_losses=policy_losses,
            value_losses=value_losses,
            entropies=entropies,
            clip_fractions=clip_fractions,
            test_rewards=test_results['rewards'])
    print(f"✓ Saved: {save_dir}/training_data.npz")

if __name__ == "__main__":
    agent, results_dir = train_ppo(n_episodes=500, update_freq=20, render_test=False)
    print(f"\n✓ Training complete! Results in: {results_dir}/")