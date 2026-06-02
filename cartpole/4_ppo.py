import csv
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

PLT_STYLE = {
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.titlesize': 14,
}
plt.rcParams.update(PLT_STYLE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LR           = 3e-4
GAMMA        = 0.99
LAM          = 0.95
CLIP_EPS     = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF   = 0.5
EPOCHS       = 10
BATCH_SIZE   = 64
UPDATE_FREQ  = 20     # episodes between policy updates
N_EPISODES   = 500
SOLVED_AVG   = 195


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),     nn.Tanh(),
            nn.Linear(hidden, action_size), nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),     nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)


class PPOAgent:
    def __init__(self, state_size, action_size):
        self.gamma   = GAMMA
        self.lam     = LAM
        self.epsilon = CLIP_EPS
        self.policy    = ActorCritic(state_size, action_size).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self._clear()

    def _clear(self):
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.values    = []
        self.log_probs = []
        self.dones     = []

    def select_action(self, state, deterministic=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs, value = self.policy(state_t)
        if deterministic:
            return probs.argmax().item(), 0.0, value.item()
        dist   = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def store(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, next_value):
        advantages, gae = [], 0
        values = self.values + [next_value]
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae   = delta + self.gamma * self.lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, self.values)]
        return advantages, returns

    def update(self):
        if not self.states:
            return 0, 0, 0, 0

        last_state = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, next_value = self.policy(last_state)

        advantages, returns = self.compute_gae(next_value.item())

        states     = torch.FloatTensor(self.states).to(DEVICE)
        actions    = torch.LongTensor(self.actions).to(DEVICE)
        old_lp     = torch.FloatTensor(self.log_probs).to(DEVICE)
        advantages = torch.FloatTensor(advantages).to(DEVICE)
        returns    = torch.FloatTensor(returns).to(DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = np.arange(len(states))
        total_pl, total_vl, total_ent, total_clip, n = 0, 0, 0, 0, 0

        for _ in range(EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, len(states), BATCH_SIZE):
                idx       = indices[start:start + BATCH_SIZE]
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
                                   torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * b_adv).mean()
                vl    = nn.MSELoss()(values.squeeze(), b_ret)
                loss  = pl + VALUE_COEF * vl - ENTROPY_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.epsilon).float().mean()

                total_pl   += pl.item()
                total_vl   += vl.item()
                total_ent  += entropy.item()
                total_clip += clip_frac.item()
                n          += 1

        self._clear()
        return total_pl / n, total_vl / n, total_ent / n, total_clip / n


def train():
    env   = gym.make('CartPole-v1')
    env.reset(seed=SEED)
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n)

    episode_rewards = []
    episode_lengths = []
    policy_losses   = []
    value_losses    = []
    entropies       = []
    clip_fractions  = []
    solve_ep        = None

    print(f"CartPole PPO | Device: {DEVICE}")
    print("=" * 50)

    for episode in range(N_EPISODES):
        state, _ = env.reset()
        total_reward, steps = 0, 0

        for _ in range(500):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(state, action, reward, log_prob, value, done)
            state        = next_state
            total_reward += reward
            steps        += 1
            if done:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if (episode + 1) % UPDATE_FREQ == 0:
            pl, vl, ent, clip = agent.update()
            policy_losses.append(pl)
            value_losses.append(vl)
            entropies.append(ent)
            clip_fractions.append(clip)

        if (episode + 1) % 25 == 0:
            avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"Episode {episode+1:4d} | Avg(100): {avg:6.2f} | Last: {total_reward:.0f}")

        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= SOLVED_AVG:
            if solve_ep is None:
                solve_ep = episode + 1
            print(f"\nSolved at episode {solve_ep}! Avg(100): {np.mean(episode_rewards[-100:]):.2f}")
            break

    env.close()
    return agent, episode_rewards, episode_lengths, policy_losses, value_losses, entropies, clip_fractions, solve_ep


def save_log(episode_rewards, episode_lengths, policy_losses, value_losses,
             entropies, clip_fractions, save_dir):
    # Per-episode log
    path = f'{save_dir}/training_log.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'steps'])
        for i in range(len(episode_rewards)):
            writer.writerow([i + 1, episode_rewards[i], episode_lengths[i]])
    print(f"Saved: {path}")

    # Per-update log
    if policy_losses:
        path2 = f'{save_dir}/update_log.csv'
        with open(path2, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['update', 'policy_loss', 'value_loss', 'entropy', 'clip_fraction'])
            for i in range(len(policy_losses)):
                writer.writerow([i + 1, round(policy_losses[i], 6), round(value_losses[i], 6),
                                 round(entropies[i], 6), round(clip_fractions[i], 6)])
        print(f"Saved: {path2}")


def save_summary(episode_rewards, episode_lengths, policy_losses, value_losses,
                 entropies, clip_fractions, solve_ep, save_dir):
    n        = len(episode_rewards)
    last_100 = episode_rewards[-100:] if n >= 100 else episode_rewards
    success  = sum(1 for r in last_100 if r >= SOLVED_AVG)

    lines = [
        "=" * 55,
        "TRAINING SUMMARY — CartPole PPO",
        "=" * 55,
        f"  Episodes trained         : {n:,}",
        f"  Policy updates           : {len(policy_losses)}",
        f"  Solved at episode        : {solve_ep if solve_ep else 'Not solved'}",
        f"  Mean reward (all)        : {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}",
        f"  Mean reward (last 100)   : {np.mean(last_100):.2f} ± {np.std(last_100):.2f}",
        f"  Best single episode      : {int(np.max(episode_rewards))}",
        f"  Success rate (last 100)  : {success} / {len(last_100)}  ({100*success/len(last_100):.1f}%)",
    ]
    if policy_losses:
        lines += [
            f"  Final policy loss        : {policy_losses[-1]:.6f}",
            f"  Final value loss         : {value_losses[-1]:.6f}",
            f"  Final entropy            : {entropies[-1]:.4f}",
            f"  Mean clip fraction       : {np.mean(clip_fractions):.3f}",
        ]
    lines += [
        "",
        "  Hyperparameters",
        f"    Learning rate          : {LR}",
        f"    Gamma / Lambda (GAE)   : {GAMMA} / {LAM}",
        f"    Clip epsilon           : {CLIP_EPS}",
        f"    Entropy coef           : {ENTROPY_COEF}",
        f"    Value coef             : {VALUE_COEF}",
        f"    Epochs per update      : {EPOCHS}",
        f"    Batch size             : {BATCH_SIZE}",
        f"    Update frequency       : every {UPDATE_FREQ} episodes",
        f"    Network                : 4 → 128 (actor) + 128 (critic)  [Tanh]",
        "=" * 55,
    ]
    text = '\n'.join(lines)
    print('\n' + text)
    with open(f'{save_dir}/summary.txt', 'w') as f:
        f.write(text + '\n')
    print(f"Saved: {save_dir}/summary.txt")


def plot_results(episode_rewards, episode_lengths, policy_losses, value_losses,
                 entropies, clip_fractions, save_dir):
    window = 100
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Raw rewards
    axes[0, 0].plot(episode_rewards, alpha=0.4, color='steelblue', linewidth=0.6, label='Per episode')
    axes[0, 0].axhline(y=SOLVED_AVG, color='red', linestyle='--', linewidth=1.5, label=f'Solved ({SOLVED_AVG})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward (= steps)')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Moving average
    if len(episode_rewards) >= window:
        ma = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        axes[0, 1].plot(range(window - 1, len(episode_rewards)), ma, color='green', linewidth=2, label=f'{window}-ep avg')
        axes[0, 1].axhline(y=SOLVED_AVG, color='red', linestyle='--', linewidth=1.5, label=f'Solved ({SOLVED_AVG})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].set_title(f'Moving Average ({window} episodes)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Policy loss
    if policy_losses:
        axes[0, 2].plot(policy_losses, color='red', linewidth=1.5, marker='o', markersize=3)
    axes[0, 2].set_xlabel('Policy Update')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('Policy Loss per Update')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Value loss
    if value_losses:
        axes[1, 0].plot(value_losses, color='purple', linewidth=1.5, marker='o', markersize=3)
    axes[1, 0].set_xlabel('Policy Update')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Value Loss per Update')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Entropy + clip fraction
    ax5 = axes[1, 1]
    if entropies:
        ax5.plot(entropies, color='green', linewidth=1.5, label='Entropy')
    ax5.set_xlabel('Policy Update')
    ax5.set_ylabel('Entropy', color='green')
    ax5.tick_params(axis='y', labelcolor='green')
    ax5b = ax5.twinx()
    if clip_fractions:
        ax5b.plot(clip_fractions, color='brown', linewidth=1.5, linestyle='--', label='Clip fraction')
        ax5b.axhline(y=0.2, color='brown', linestyle=':', alpha=0.5)
    ax5b.set_ylabel('Clip Fraction', color='brown')
    ax5b.tick_params(axis='y', labelcolor='brown')
    ax5.set_title('Entropy & Clip Fraction per Update')
    ax5.grid(True, alpha=0.3)

    # 6. Reward distribution
    axes[1, 2].hist(episode_rewards, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    axes[1, 2].axvline(x=SOLVED_AVG, color='red', linestyle='--', linewidth=1.5, label=f'Solved ({SOLVED_AVG})')
    axes[1, 2].axvline(x=np.mean(episode_rewards), color='orange', linewidth=1.5,
                        label=f'Mean: {np.mean(episode_rewards):.1f}')
    axes[1, 2].set_xlabel('Reward')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Reward Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('CartPole PPO — Training Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/training_results.png")


def test_agent(agent, n_episodes=20):
    env = gym.make('CartPole-v1')
    test_rewards = []

    print(f"\nTesting trained agent ({n_episodes} episodes)...")
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(500):
            action, _, _ = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        test_rewards.append(total_reward)
        status = "✓" if total_reward >= SOLVED_AVG else "✗"
        print(f"  {status} Test {episode+1:2d}: {total_reward:.0f} steps")

    avg          = np.mean(test_rewards)
    success_rate = 100 * np.sum(np.array(test_rewards) >= SOLVED_AVG) / n_episodes
    print(f"\n  Average : {avg:.2f} ± {np.std(test_rewards):.2f}")
    print(f"  Success : {success_rate:.0f}%")
    print(f"  Result  : {'SOLVED' if avg >= SOLVED_AVG else 'Needs more training'}")
    env.close()


if __name__ == "__main__":
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/cartpole_ppo_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    agent, rewards, lengths, pl, vl, ent, clip, solve_ep = train()

    save_log(rewards, lengths, pl, vl, ent, clip, results_dir)
    save_summary(rewards, lengths, pl, vl, ent, clip, solve_ep, results_dir)
    plot_results(rewards, lengths, pl, vl, ent, clip, results_dir)
    test_agent(agent)

    torch.save(agent.policy.state_dict(), f'{results_dir}/model.pth')
    print(f"\nResults saved to: {results_dir}/")
