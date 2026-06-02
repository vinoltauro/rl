import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from datetime import datetime
import os
import gymnasium as gym

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LR            = 3e-4
GAMMA         = 0.99
LAM           = 0.95
CLIP_EPS      = 0.2
ENTROPY_COEF  = 0.05
VALUE_COEF    = 0.5
EPOCHS        = 10
BATCH_SIZE    = 64
UPDATE_FREQ   = 2048   # steps between updates
N_EPISODES    = 1000
MAX_STEPS     = 200
SOLVED_AVG    = -110.0


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),     nn.Tanh(),
            nn.Linear(hidden, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),     nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state):
        logits = self.actor(state)
        probs  = torch.softmax(logits, dim=-1)
        value  = self.critic(state)
        return probs, value


class PPOAgent:
    def __init__(self, state_size, action_size):
        self.policy    = ActorCritic(state_size, action_size).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR, eps=1e-5)
        self._clear()

    def _clear(self):
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.values    = []
        self.log_probs = []
        self.dones     = []

    def normalize_state(self, state):
        s = np.array(state, dtype=np.float32)
        s[0] = (s[0] + 0.3) / 0.9
        s[1] = s[1] / 0.07
        return s

    def select_action(self, state, deterministic=False):
        s = torch.FloatTensor(self.normalize_state(state)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs, value = self.policy(s)
        if deterministic:
            return probs.argmax().item(), 0.0, 0.0
        dist   = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def store(self, state, action, reward, log_prob, value, done):
        self.states.append(self.normalize_state(state))
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def update(self):
        if not self.states:
            return 0, 0, 0

        last_s = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, next_value = self.policy(last_s)
            next_value = next_value.item()

        # GAE
        advantages, gae = [], 0
        values = self.values + [next_value]
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + GAMMA * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae   = delta + GAMMA * LAM * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)

        states     = torch.FloatTensor(np.array(self.states)).to(DEVICE)
        actions    = torch.LongTensor(self.actions).to(DEVICE)
        old_lp     = torch.FloatTensor(self.log_probs).to(DEVICE)
        advantages = torch.FloatTensor(advantages).to(DEVICE)
        returns    = advantages + torch.FloatTensor(self.values).to(DEVICE)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(len(states))
        total_pl, total_vl, total_ent, n = 0, 0, 0, 0

        for _ in range(EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, len(states), BATCH_SIZE):
                idx = indices[start:start + BATCH_SIZE]
                b_states  = states[idx]
                b_actions = actions[idx]
                b_old_lp  = old_lp[idx]
                b_adv     = advantages[idx]
                b_ret     = returns[idx]

                probs, values = self.policy(b_states)
                dist          = Categorical(probs)
                new_lp        = dist.log_prob(b_actions)
                entropy       = dist.entropy().mean()

                ratio = torch.exp(new_lp - b_old_lp)
                pl    = -torch.min(ratio * b_adv,
                                   torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * b_adv).mean()
                vl    = 0.5 * ((values.squeeze() - b_ret) ** 2).mean()
                loss  = pl + VALUE_COEF * vl - ENTROPY_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_pl  += pl.item()
                total_vl  += vl.item()
                total_ent += entropy.item()
                n         += 1

        self._clear()
        return total_pl / n, total_vl / n, total_ent / n


def shaped_reward(position, velocity, terminated):
    height = (position + 1.2) / 1.8 * 2.0
    ke     = velocity * velocity
    reward = height + 100.0 * ke - 1.0
    if terminated and position >= 0.5:
        reward += 10.0
    return reward


def train():
    env   = gym.make('MountainCar-v0')
    env.reset(seed=SEED)
    agent = PPOAgent(2, env.action_space.n)

    episode_rewards     = []
    episode_raw_rewards = []
    episode_lengths     = []
    policy_losses       = []
    value_losses        = []
    entropies           = []

    print(f"MountainCar PPO | Device: {DEVICE}")
    print("=" * 50)

    for episode in range(N_EPISODES):
        state, _ = env.reset()
        ep_reward     = 0
        ep_raw_reward = 0
        steps         = 0

        for _ in range(MAX_STEPS):
            action, log_prob, val = agent.select_action(state)
            next_state, raw_reward, terminated, truncated, _ = env.step(action)
            done  = terminated or truncated
            s_rew = shaped_reward(next_state[0], next_state[1], terminated)

            agent.store(state, action, s_rew, log_prob, val, done)
            state          = next_state
            ep_reward     += s_rew
            ep_raw_reward += raw_reward
            steps         += 1
            if len(agent.states) >= UPDATE_FREQ:
                pl, vl, ent = agent.update()
                policy_losses.append(pl)
                value_losses.append(vl)
                entropies.append(ent)

            if done:
                break

        episode_rewards.append(ep_reward)
        episode_raw_rewards.append(ep_raw_reward)
        episode_lengths.append(steps)

        if (episode + 1) % 50 == 0:
            avg_raw = np.mean(episode_raw_rewards[-100:]) if len(episode_raw_rewards) >= 100 else np.mean(episode_raw_rewards)
            success = sum(1 for r in episode_raw_rewards[-100:] if r >= SOLVED_AVG)
            print(f"Episode {episode+1:4d} | Avg raw(100): {avg_raw:7.2f} | Success: {success:3d}/100 | Steps: {steps:3d}")

        if steps < MAX_STEPS:
            print(f"  >>> Episode {episode+1} REACHED GOAL in {steps} steps!")

        if len(episode_raw_rewards) >= 100 and np.mean(episode_raw_rewards[-100:]) >= SOLVED_AVG:
            print(f"\nSolved at episode {episode+1}! Avg raw(100): {np.mean(episode_raw_rewards[-100:]):.2f}")
            break

    env.close()
    return agent, episode_raw_rewards, episode_lengths, policy_losses, value_losses, entropies


def plot_results(episode_raw_rewards, episode_lengths, policy_losses,
                 value_losses, entropies, save_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    window = 100

    axes[0, 0].plot(episode_raw_rewards, alpha=0.5, color='blue', linewidth=0.5)
    axes[0, 0].axhline(y=SOLVED_AVG, color='red', linestyle='--', label=f'Solved ({SOLVED_AVG})')
    axes[0, 0].set_title('Episode Rewards (real score)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    if len(episode_raw_rewards) >= window:
        ma = np.convolve(episode_raw_rewards, np.ones(window) / window, mode='valid')
        axes[0, 1].plot(range(window - 1, len(episode_raw_rewards)), ma, color='green', linewidth=2)
        axes[0, 1].axhline(y=SOLVED_AVG, color='red', linestyle='--')
    axes[0, 1].set_title(f'Moving Average ({window} episodes)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(episode_lengths, alpha=0.5, color='orange', linewidth=0.5)
    axes[0, 2].axhline(y=110, color='green', linestyle='--', label='Target (<110 steps)')
    axes[0, 2].set_title('Episode Lengths')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Steps')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    if policy_losses:
        axes[1, 0].plot(policy_losses, color='red', linewidth=1.5)
    axes[1, 0].set_title('Policy Loss')
    axes[1, 0].set_xlabel('Update')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)

    if value_losses:
        axes[1, 1].plot(value_losses, color='purple', linewidth=1.5)
    axes[1, 1].set_title('Value Loss')
    axes[1, 1].set_xlabel('Update')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)

    if entropies:
        axes[1, 2].plot(entropies, color='green', linewidth=1.5)
    axes[1, 2].set_title('Policy Entropy')
    axes[1, 2].set_xlabel('Update')
    axes[1, 2].set_ylabel('Entropy')
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle('MountainCar PPO', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/training_results.png")


def test_agent(agent, n_episodes=20):
    env = gym.make('MountainCar-v0')
    test_rewards = []

    print(f"\nTesting trained agent ({n_episodes} episodes)...")
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        for _ in range(MAX_STEPS):
            action, _, _ = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        test_rewards.append(total_reward)
        status = "✓" if total_reward >= SOLVED_AVG else "✗"
        print(f"  {status} Test {episode+1:2d}: {total_reward:6.1f}  ({steps:3d} steps)")

    avg     = np.mean(test_rewards)
    success = sum(1 for r in test_rewards if r >= SOLVED_AVG)
    print(f"\n  Average: {avg:.2f}  |  Success: {success}/{n_episodes}")
    print(f"  {'Solved!' if avg >= SOLVED_AVG else 'Needs more training.'}")
    env.close()


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/mountaincar_ppo_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    agent, rewards, lengths, pl, vl, ent = train()
    plot_results(rewards, lengths, pl, vl, ent, results_dir)
    test_agent(agent)

    torch.save(agent.policy.state_dict(), f'{results_dir}/model.pth')
    print(f"\nResults saved to: {results_dir}/")
