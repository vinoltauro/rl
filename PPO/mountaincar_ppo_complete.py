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
    def __init__(self, state_size, action_size, hidden_size=64):
        super(ActorCritic, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, state):
        action_logits = self.actor(state)
        action_probs = torch.softmax(action_logits, dim=-1)
        state_value = self.critic(state)
        return action_probs, state_value

class PPOAgent:
    def __init__(self, state_size, action_size, hidden_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.lam = 0.95
        self.epsilon = 0.2
        self.lr = 3e-4
        self.epochs = 10
        self.batch_size = 64
        self.entropy_coef = 0.05
        self.value_coef = 0.5
        self.update_frequency = 2048
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, eps=1e-5)
        
        self.clear_memory()
        
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def normalize_state(self, state):
        state_norm = np.array(state, dtype=np.float32)
        state_norm[0] = (state_norm[0] + 0.3) / 0.9
        state_norm[1] = state_norm[1] / 0.07
        return state_norm

    def select_action(self, state, deterministic=False):
        state_norm = self.normalize_state(state)
        state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, state_value = self.policy(state_t)
        
        if deterministic:
            return action_probs.argmax().item(), 0.0, 0.0
            
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), state_value.item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        state_norm = self.normalize_state(state)
        self.states.append(state_norm)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def update(self):
        if len(self.states) == 0:
            return 0, 0, 0, 0

        last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, next_value = self.policy(last_state)
            next_value = next_value.item()
            
        advantages = []
        gae = 0
        values = self.values + [next_value]
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(self.values).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        steps = 0
        
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                
                new_action_probs, new_state_values = self.policy(batch_states)
                dist = Categorical(new_action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = 0.5 * ((new_state_values.squeeze() - batch_returns) ** 2).mean()
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                steps += 1
                
        self.clear_memory()
        return total_policy_loss/steps, total_value_loss/steps, total_entropy/steps, 0

def shaped_reward(position, velocity, done):
    height = (position + 1.2) / 1.8 * 2.0
    ke = velocity * velocity
    reward = height + (100 * ke) - 1.0
    
    if done and position >= 0.5:
        reward += 10.0
        
    return reward

def train_ppo(n_episodes=1000):
    env = gym.make('MountainCar-v0')
    agent = PPOAgent(2, 3, hidden_size=64)
    
    episode_rewards = []
    episode_raw_rewards = []
    episode_lengths = []
    policy_losses = []
    value_losses = []
    entropies = []
    success_count = 0
    
    print("="*70)
    print("MOUNTAINCAR PPO - PROPER IMPLEMENTATION")
    print("="*70)
    print(f"Device: {agent.device}")
    print("Key Features:")
    print("  • Orthogonal initialization")
    print("  • Input normalization")
    print("  • Physics-based reward shaping")
    print("  • Batch updates every 2048 steps")
    print("  • Smaller network (64 units)")
    print("="*70)
    
    total_steps = 0
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_raw_reward = 0
        steps = 0
        
        for step in range(200):
            action, log_prob, val = agent.select_action(state)
            next_state, raw_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            shaped_rew = shaped_reward(next_state[0], next_state[1], terminated)
            
            agent.store_transition(state, action, shaped_rew, log_prob, val, done)
            
            state = next_state
            episode_reward += shaped_rew
            episode_raw_reward += raw_reward
            steps += 1
            total_steps += 1
            
            if len(agent.states) >= agent.update_frequency:
                pl, vl, ent, _ = agent.update()
                policy_losses.append(pl)
                value_losses.append(vl)
                entropies.append(ent)
            
            if done:
                if terminated:
                    success_count += 1
                break
        
        episode_rewards.append(episode_reward)
        episode_raw_rewards.append(episode_raw_reward)
        episode_lengths.append(steps)
        
        if (episode + 1) % 25 == 0:
            avg_raw = np.mean(episode_raw_rewards[-100:]) if len(episode_raw_rewards) >= 100 else np.mean(episode_raw_rewards)
            recent_success = np.sum([1 for r in episode_raw_rewards[-100:] if r > -200])
            print(f"Ep {episode+1:4d} | Steps: {steps:3d} | Raw: {episode_raw_reward:6.1f} | Avg: {avg_raw:7.2f} | Success: {recent_success:.0f}/100")
        
        if steps < 199:
            print(f"  >>> Episode {episode+1} SOLVED! (Reached goal in {steps} steps)")
        
        if len(episode_raw_rewards) >= 100 and np.mean(episode_raw_rewards[-100:]) >= -110:
            print(f"\n{'='*70}")
            print(f"🎉 SOLVED at episode {episode + 1}!")
            print(f"Average reward: {np.mean(episode_raw_rewards[-100:]):.2f}")
            print(f"{'='*70}")
            break
    
    env.close()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"mountaincar_ppo_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    create_plots(episode_raw_rewards, episode_lengths, policy_losses, 
                 value_losses, entropies, results_dir)
    
    test_results = test_agent(agent, save_dir=results_dir)
    
    torch.save(agent.policy.state_dict(), f'{results_dir}/ppo_model.pth')
    print(f"✓ Saved model: {results_dir}/ppo_model.pth")
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_dir}/")
    print(f"{'='*70}")
    
    return agent, results_dir

def create_plots(episode_rewards, episode_lengths, policy_losses, 
                 value_losses, entropies, save_dir):
    
    fig = plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(episode_rewards, alpha=0.6, color='blue', linewidth=0.8)
    plt.axhline(y=-110, color='red', linestyle='--', linewidth=2, label='Target (-110)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    if len(episode_rewards) >= 100:
        ma = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        plt.plot(range(99, len(episode_rewards)), ma, color='green', linewidth=2)
        plt.axhline(y=-110, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Moving Average (100)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.plot(episode_lengths, alpha=0.6, color='orange', linewidth=0.8)
    plt.axhline(y=110, color='green', linestyle='--', linewidth=2, label='Target')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episode Lengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 4)
    if policy_losses:
        plt.plot(policy_losses, color='red', linewidth=1.5)
    plt.xlabel('Update')
    plt.ylabel('Loss')
    plt.title('Policy Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    if value_losses:
        plt.plot(value_losses, color='purple', linewidth=1.5)
    plt.xlabel('Update')
    plt.ylabel('Loss')
    plt.title('Value Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    if entropies:
        plt.plot(entropies, color='green', linewidth=1.5)
    plt.xlabel('Update')
    plt.ylabel('Entropy')
    plt.title('Policy Entropy')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('MountainCar PPO Training Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_results.png', dpi=150)
    print(f"✓ Saved: {save_dir}/training_results.png")
    plt.close()

def test_agent(agent, n_episodes=20, save_dir=None):
    env = gym.make('MountainCar-v0')
    test_rewards = []
    test_lengths = []
    
    print(f"\n{'='*70}")
    print("TESTING AGENT")
    print(f"{'='*70}")
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(200):
            action, _, _ = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        test_rewards.append(total_reward)
        test_lengths.append(steps)
        status = "✓" if total_reward > -200 else "✗"
        print(f"{status} Test {episode+1:2d}: {total_reward:6.1f} ({steps:3d} steps)")
    
    avg = np.mean(test_rewards)
    success_rate = 100 * np.sum(np.array(test_rewards) > -200) / n_episodes
    
    print(f"\n{'='*70}")
    print(f"Average: {avg:.2f}")
    print(f"Success Rate: {success_rate:.0f}%")
    if avg >= -110:
        print("🎉 SOLVED!")
    elif avg >= -130:
        print("⚠ Close!")
    print(f"{'='*70}")
    
    env.close()
    
    return {
        'rewards': test_rewards,
        'lengths': test_lengths,
        'mean': avg,
        'success_rate': success_rate
    }

if __name__ == "__main__":
    agent, results_dir = train_ppo(n_episodes=500)
    print(f"\n✓ Complete! Results in: {results_dir}/")