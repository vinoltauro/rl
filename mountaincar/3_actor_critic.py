import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import os
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

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

# Hyperparameters — vectorized A2C with n_envs=16
# SB3 Zoo requires n_envs=16 for MountainCar; this replicates that setup.
N_ENVS       = 16     # parallel environments
N_STEPS      = 16     # steps collected per env before each update (256 total per update)
N_UPDATES    = 4000   # total parameter updates = 4000 × 256 = 1,024,000 env steps
ACTOR_LR     = 3e-4
CRITIC_LR    = 1e-4
VALUE_COEF   = 0.5
ENTROPY_COEF = 0.02
GAMMA        = 0.99
LAM          = 0.95
MAX_STEPS    = 200
SOLVED_AVG   = -110.0


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

    def forward(self, x):
        return torch.softmax(self.actor(x), dim=-1), self.critic(x)


def normalize_state(state):
    """Works for both (2,) and (N, 2) shaped inputs."""
    s = np.array(state, dtype=np.float32).copy()
    if s.ndim == 1:
        s[0] = (s[0] + 0.3) / 0.9
        s[1] = s[1] / 0.07
    else:
        s[:, 0] = (s[:, 0] + 0.3) / 0.9
        s[:, 1] = s[:, 1] / 0.07
    return s


def shaped_reward(position, velocity, terminated):
    height = (position + 1.2) / 1.8 * 2.0
    ke     = velocity * velocity
    reward = height + 100.0 * ke - 1.0
    if terminated and position >= 0.5:
        reward += 10.0
    return reward


def train():
    def make_env():
        return gym.make('MountainCar-v0', max_episode_steps=MAX_STEPS)

    envs  = SyncVectorEnv([make_env for _ in range(N_ENVS)])
    model = ActorCritic(2, envs.single_action_space.n).to(DEVICE)
    optimizer = optim.Adam([
        {'params': model.actor.parameters(),  'lr': ACTOR_LR},
        {'params': model.critic.parameters(), 'lr': CRITIC_LR},
    ], eps=1e-5)

    obs, _ = envs.reset(seed=list(range(SEED, SEED + N_ENVS)))

    ep_raw      = np.zeros(N_ENVS)          # accumulate raw reward per env
    all_rewards = []                         # completed episode raw rewards
    goal_count  = 0
    solve_ep    = None
    actor_losses, critic_losses = [], []

    print(f"MountainCar A2C (vectorized n_envs={N_ENVS}) | Device: {DEVICE}")
    print(f"Total env steps: {N_UPDATES * N_STEPS * N_ENVS:,}")
    print("=" * 55)

    for update in range(N_UPDATES):
        # ── collect N_STEPS steps from all N_ENVS environments ──────────────
        mb_obs   = np.zeros((N_STEPS, N_ENVS, 2),   dtype=np.float32)
        mb_act   = np.zeros((N_STEPS, N_ENVS),       dtype=np.int64)
        mb_rew   = np.zeros((N_STEPS, N_ENVS),       dtype=np.float32)
        mb_val   = np.zeros((N_STEPS, N_ENVS),       dtype=np.float32)
        mb_lp    = np.zeros((N_STEPS, N_ENVS),       dtype=np.float32)
        mb_done  = np.zeros((N_STEPS, N_ENVS),       dtype=np.float32)
        mb_term  = np.zeros((N_STEPS, N_ENVS),       dtype=np.float32)

        for step in range(N_STEPS):
            obs_norm = normalize_state(obs)
            obs_t    = torch.FloatTensor(obs_norm).to(DEVICE)

            with torch.no_grad():
                probs, vals = model(obs_t)
            dist    = torch.distributions.Categorical(probs)
            actions = dist.sample()

            next_obs, raw_rew, terminated, truncated, info = envs.step(
                actions.cpu().numpy()
            )
            done = terminated | truncated

            # shaped reward: use next_obs (post-step) position and velocity
            s_rew = np.array([
                shaped_reward(next_obs[i, 0], next_obs[i, 1], terminated[i])
                for i in range(N_ENVS)
            ], dtype=np.float32)

            mb_obs[step]  = obs
            mb_act[step]  = actions.cpu().numpy()
            mb_rew[step]  = s_rew
            mb_val[step]  = vals.squeeze(-1).cpu().numpy()
            mb_lp[step]   = dist.log_prob(actions).cpu().numpy()
            mb_done[step] = done.astype(np.float32)
            mb_term[step] = terminated.astype(np.float32)

            ep_raw += raw_rew
            for i in range(N_ENVS):
                if done[i]:
                    all_rewards.append(float(ep_raw[i]))
                    if ep_raw[i] > -199.5:
                        goal_count += 1
                    ep_raw[i] = 0.0

            # When an env auto-resets, use final_observation for bootstrap
            # so we don't bootstrap on the reset state for truncated episodes.
            # gymnasium vectorized envs store this in info['final_observation'].
            for i in range(N_ENVS):
                if truncated[i] and not terminated[i]:
                    if 'final_observation' in info and info['final_observation'][i] is not None:
                        final_s = normalize_state(info['final_observation'][i])
                        final_t = torch.FloatTensor(final_s).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            _, fv = model(final_t)
                        # Patch: mark this env's last value as the true final V
                        mb_val[step, i] = fv.item()

            obs = next_obs

        # ── bootstrap value for the state we're leaving ─────────────────────
        obs_norm   = normalize_state(obs)
        obs_t      = torch.FloatTensor(obs_norm).to(DEVICE)
        with torch.no_grad():
            _, final_vals = model(obs_t)
        next_v = final_vals.squeeze(-1).cpu().numpy()

        # ── GAE across all envs, respecting episode boundaries ───────────────
        advantages = np.zeros((N_STEPS, N_ENVS), dtype=np.float32)
        gae        = np.zeros(N_ENVS, dtype=np.float32)

        for t in reversed(range(N_STEPS)):
            not_terminal = 1.0 - mb_term[t]   # V=0 only at true goal terminals
            not_done     = 1.0 - mb_done[t]   # GAE propagation stops at episode end
            delta  = mb_rew[t] + GAMMA * next_v * not_terminal - mb_val[t]
            gae    = delta + GAMMA * LAM * not_done * gae
            advantages[t] = gae
            next_v = mb_val[t]

        returns = advantages + mb_val

        # ── single gradient update on the full (N_STEPS × N_ENVS) batch ─────
        flat_obs = normalize_state(mb_obs.reshape(-1, 2))
        flat_act = mb_act.reshape(-1)
        flat_adv = advantages.reshape(-1)
        flat_ret = returns.reshape(-1)

        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        obs_t  = torch.FloatTensor(flat_obs).to(DEVICE)
        act_t  = torch.LongTensor(flat_act).to(DEVICE)
        adv_t  = torch.FloatTensor(flat_adv).to(DEVICE)
        ret_t  = torch.FloatTensor(flat_ret).to(DEVICE)

        probs, vals = model(obs_t)
        dist     = torch.distributions.Categorical(probs)
        new_lp   = dist.log_prob(act_t)
        entropy  = dist.entropy().mean()

        policy_loss = -(new_lp * adv_t).mean()
        value_loss  = F.smooth_l1_loss(vals.squeeze(-1), ret_t)
        loss        = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        actor_losses.append(policy_loss.item())
        critic_losses.append(value_loss.item())

        if (update + 1) % 200 == 0:
            n_ep = len(all_rewards)
            avg  = np.mean(all_rewards[-100:]) if n_ep >= 100 else (np.mean(all_rewards) if all_rewards else -200.0)
            succ = sum(1 for r in all_rewards[-100:] if r >= SOLVED_AVG) if n_ep >= 100 else 0
            steps = (update + 1) * N_STEPS * N_ENVS
            print(f"Update {update+1:4d} | Steps: {steps:8,d} | Episodes: {n_ep:5,d} | "
                  f"Avg(100): {avg:7.2f} | Goals: {goal_count:4d} | Solved: {succ}/100")

            if goal_count > 0 and solve_ep is None:
                solve_ep = n_ep

        if len(all_rewards) >= 100 and np.mean(all_rewards[-100:]) >= SOLVED_AVG:
            n_ep = len(all_rewards)
            if solve_ep is None:
                solve_ep = n_ep
            print(f"\nSolved at episode {solve_ep}! Avg(100): {np.mean(all_rewards[-100:]):.2f}")
            break

    envs.close()
    return model, all_rewards, goal_count, actor_losses, critic_losses, solve_ep


def save_log(all_rewards, actor_losses, critic_losses, save_dir):
    path = f'{save_dir}/training_log.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'raw_reward', 'actor_loss', 'critic_loss'])
        for i, r in enumerate(all_rewards):
            al = actor_losses[i] if i < len(actor_losses) else 0.0
            cl = critic_losses[i] if i < len(critic_losses) else 0.0
            writer.writerow([i + 1, r, round(al, 6), round(cl, 6)])
    print(f"Saved: {path}")


def save_summary(all_rewards, goal_count, solve_ep, save_dir):
    n        = len(all_rewards)
    last_100 = all_rewards[-100:] if n >= 100 else all_rewards
    success  = sum(1 for r in last_100 if r >= SOLVED_AVG)
    lines = [
        "=" * 55,
        "TRAINING SUMMARY — MountainCar A2C (vectorized)",
        "=" * 55,
        f"  N_ENVS / N_STEPS            : {N_ENVS} / {N_STEPS}",
        f"  Total env steps              : {N_UPDATES * N_STEPS * N_ENVS:,}",
        f"  Total episodes (all envs)    : {n:,}",
        f"  Solved at episode            : {solve_ep if solve_ep else 'Not solved'}",
        f"  Mean reward (all)            : {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}",
        f"  Mean reward (last 100)       : {np.mean(last_100):.2f} ± {np.std(last_100):.2f}",
        f"  Success rate (last 100)      : {success} / {len(last_100)}  ({100*success/len(last_100):.1f}%)",
        f"  Total goals reached          : {goal_count:,}",
        "",
        "  Hyperparameters",
        f"    Actor LR / Critic LR       : {ACTOR_LR} / {CRITIC_LR}",
        f"    Value coef / Entropy coef  : {VALUE_COEF} / {ENTROPY_COEF}",
        f"    Gamma / Lambda (GAE)       : {GAMMA} / {LAM}",
        f"    Gradient clip norm         : 0.5",
        f"    Network hidden             : 64 units, 2 layers (Tanh)",
        f"    Initialisation             : orthogonal",
        f"    State normalisation        : pos [-1,1], vel [-1,1]",
        f"    Reward shaping             : height + 100·KE - 1  (+10 at goal)",
        "=" * 55,
    ]
    text = '\n'.join(lines)
    print('\n' + text)
    with open(f'{save_dir}/summary.txt', 'w') as f:
        f.write(text + '\n')


def plot_results(all_rewards, actor_losses, critic_losses, save_dir):
    window = 100
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    eps = np.arange(1, len(all_rewards) + 1)
    axes[0, 0].plot(eps, all_rewards, alpha=0.3, color='steelblue', lw=0.5)
    if len(all_rewards) >= window:
        ma = np.convolve(all_rewards, np.ones(window) / window, mode='valid')
        axes[0, 0].plot(range(window, len(all_rewards) + 1), ma, color='navy', lw=2, label=f'{window}-ep avg')
    axes[0, 0].axhline(y=SOLVED_AVG, color='red', ls='--', lw=1.5, label=f'Solved ({SOLVED_AVG})')
    axes[0, 0].set_xlabel('Episode'); axes[0, 0].set_ylabel('Reward (real score)')
    axes[0, 0].set_title('Episode Rewards'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    if len(all_rewards) >= window:
        axes[0, 1].plot(range(window, len(all_rewards) + 1), ma, color='green', lw=2)
        axes[0, 1].axhline(y=SOLVED_AVG, color='red', ls='--', lw=1.5)
    axes[0, 1].set_xlabel('Episode'); axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].set_title(f'Moving Average ({window} episodes)'); axes[0, 1].grid(True, alpha=0.3)

    if actor_losses:
        upd = np.arange(1, len(actor_losses) + 1)
        axes[1, 0].plot(upd, np.abs(actor_losses), color='darkorange', lw=1.0, alpha=0.7, label='|Actor loss|')
        axes[1, 0].plot(upd, critic_losses,          color='teal',       lw=1.5, label='Critic loss')
        axes[1, 0].set_yscale('symlog', linthresh=1e-4)
        axes[1, 0].set_xlabel('Update'); axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Actor and Critic Loss per Update')
        axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(all_rewards, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    axes[1, 1].axvline(x=SOLVED_AVG, color='red', ls='--', lw=1.5, label=f'Solved ({SOLVED_AVG})')
    axes[1, 1].axvline(x=np.mean(all_rewards), color='orange', lw=1.5,
                        label=f'Mean: {np.mean(all_rewards):.1f}')
    axes[1, 1].set_xlabel('Reward'); axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Distribution'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'MountainCar A2C (n_envs={N_ENVS}) — Training Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/training_results.png")


def plot_trajectory(model, save_dir):
    env = gym.make('MountainCar-v0')
    state, _ = env.reset(seed=0)
    positions, velocities = [], []

    for _ in range(MAX_STEPS):
        positions.append(state[0])
        velocities.append(state[1])
        s_norm = normalize_state(state)
        s_t    = torch.from_numpy(s_norm).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs, _ = model(s_t)
        action = probs.argmax().item()
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    positions.append(state[0])
    velocities.append(state[1])
    env.close()

    reached = max(positions) >= 0.5
    status  = f"REACHED GOAL in {len(positions)-1} steps" if reached else f"DID NOT REACH GOAL ({len(positions)-1} steps)"

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(range(len(positions)), positions, color='steelblue', lw=1.8)
    axes[0].axhline(y=0.5,  color='green', ls='--', lw=1.5, label='Goal (pos = 0.5)')
    axes[0].axhline(y=-0.5, color='gray',  ls=':',  alpha=0.5, label='Valley bottom')
    axes[0].set_xlabel('Timestep'); axes[0].set_ylabel('Position')
    axes[0].set_title('Car Position Over Time'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(len(velocities)), velocities, color='darkorange', lw=1.8)
    axes[1].axhline(y=0, color='gray', ls='--', alpha=0.5, label='Zero velocity')
    axes[1].set_xlabel('Timestep'); axes[1].set_ylabel('Velocity')
    axes[1].set_title('Car Velocity Over Time'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Test Episode Trajectory — {status}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir}/trajectory.png")


def test_agent(model, n_episodes=20):
    env = gym.make('MountainCar-v0')
    test_rewards = []
    model.eval()
    print(f"\nTesting trained agent ({n_episodes} episodes)...")
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        for _ in range(MAX_STEPS):
            s_norm = normalize_state(state)
            s_t    = torch.from_numpy(s_norm).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs, _ = model(s_t)
            action = probs.argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        test_rewards.append(total_reward)
        status = "✓" if total_reward >= SOLVED_AVG else "✗"
        print(f"  {status} Test {episode+1:2d}: {total_reward:6.1f}")
    avg     = np.mean(test_rewards)
    success = sum(1 for r in test_rewards if r >= SOLVED_AVG)
    print(f"\n  Average : {avg:.2f} ± {np.std(test_rewards):.2f}")
    print(f"  Success : {success}/{n_episodes}  ({100*success/n_episodes:.0f}%)")
    print(f"  Result  : {'SOLVED' if avg >= SOLVED_AVG else 'Not solved'}")
    env.close()


if __name__ == "__main__":
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/mountaincar_ac_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    model, all_rewards, goal_count, a_losses, c_losses, solve_ep = train()

    save_log(all_rewards, a_losses, c_losses, results_dir)
    save_summary(all_rewards, goal_count, solve_ep, results_dir)
    plot_results(all_rewards, a_losses, c_losses, results_dir)
    plot_trajectory(model, results_dir)
    test_agent(model)

    torch.save(model.state_dict(), f'{results_dir}/model.pth')
    print(f"\nResults saved to: {results_dir}/")
