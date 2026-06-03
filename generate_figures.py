"""
Generate all dissertation figures from existing training CSVs.
Run once: python3 generate_figures.py
Saves to latex/figures/ for direct inclusion in report.tex
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs('latex/figures', exist_ok=True)

PLT = {
    'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.titlesize': 14,
}
plt.rcParams.update(PLT)
BLUE = '#0569B9'  # TCD blue

def moving_avg(data, w):
    return np.convolve(data, np.ones(w)/w, mode='valid')

def read_csv(path, col):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [float(r[col]) for r in rows]

# ── helpers ─────────────────────────────────────────────────────────────────

def save(name):
    path = f'latex/figures/{name}.pdf'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.savefig(path.replace('.pdf','.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {path}')

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — CartPole: all 4 algorithms on one plot
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))

runs = [
    ('Q-Learning',    'results/cartpole_qlearning_20260602_103536/training_log.csv', 'reward', 100, '#e41a1c'),
    ('DQN',           'results/cartpole_dqn_20260602_110126/training_log.csv',        'reward', 30,  '#377eb8'),
    ('A2C',           'results/cartpole_ac_20260602_110127/training_log.csv',          'reward', 50, '#4daf4a'),
    ('PPO',           'results/cartpole_ppo_20260602_103540/training_log.csv',         'reward', 20, '#ff7f00'),
]

solve_eps = {'Q-Learning': 9798, 'DQN': 225, 'A2C': 1175, 'PPO': 195}

for label, path, col, w, color in runs:
    try:
        rewards = read_csv(path, col)
        ma = moving_avg(rewards, w)
        x = range(w-1, len(rewards))
        ax.plot(x, ma, label=label, color=color, linewidth=2)
        sol = solve_eps[label]
        ax.axvline(x=sol, color=color, linestyle='--', alpha=0.5, linewidth=1)
    except Exception as e:
        print(f'  skip {label}: {e}')

ax.axhline(y=195, color='black', linestyle=':', linewidth=1.5, label='Solved threshold (195)')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward (moving average)')
ax.set_title('CartPole-v1: All Four Algorithms')
ax.legend(loc='upper left')
ax.set_xlim(0, 10000)
ax.set_ylim(0, 520)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save('cartpole_all_algorithms')

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — GridWorld training curve
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

rewards = read_csv('results/gridworld_qlearning_20260602_103536/training_log.csv', 'reward')
ma = moving_avg(rewards, 50)
axes[0].plot(rewards, alpha=0.3, color=BLUE, linewidth=0.5)
axes[0].plot(range(49, len(rewards)), ma, color=BLUE, linewidth=2, label='50-ep avg')
axes[0].axhline(y=9, color='green', linestyle='--', linewidth=1.5, label='Near-optimal (9)')
axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Reward')
axes[0].set_title('GridWorld Q-Learning: Reward')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

try:
    steps = read_csv('results/gridworld_qlearning_20260602_103536/training_log.csv', 'steps')
    ma_s = moving_avg(steps, 50)
    axes[1].plot(steps, alpha=0.3, color='#ff7f00', linewidth=0.5)
    axes[1].plot(range(49, len(steps)), ma_s, color='#ff7f00', linewidth=2)
    axes[1].axhline(y=10, color='green', linestyle='--', linewidth=1.5, label='Optimal (10 steps)')
    axes[1].set_xlabel('Episode'); axes[1].set_ylabel('Steps to goal')
    axes[1].set_title('GridWorld Q-Learning: Episode Length')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
except: pass

plt.suptitle('GridWorld Q-Learning Training', fontsize=14, fontweight='bold')
plt.tight_layout()
save('gridworld_training')

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — MountainCar: best run per algorithm comparison
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))

mc_runs = [
    ('Q-Learning (raw, 50K eps)',
     'results/mountaincar_qlearning_20260602_110124/training_log.csv',
     'reward', 500, '#e41a1c'),
    ('Double DQN (v3, 5K eps)',
     'results/mountaincar_dqn_20260602_171518/training_log.csv',
     'reward', 100, '#377eb8'),
    ('PPO v5 (8K eps)',
     'results/mountaincar_ppo_20260602_195214/training_log.csv',
     'raw_reward', 100, '#ff7f00'),
]

for label, path, col, w, color in mc_runs:
    try:
        rewards = read_csv(path, col)
        ma = moving_avg(rewards, w)
        x_max = min(len(rewards), 10000)
        x = range(w-1, x_max)
        ax.plot(list(x), ma[:len(x)], label=label, color=color, linewidth=2)
    except Exception as e:
        print(f'  skip {label}: {e}')

ax.axhline(y=-110, color='black', linestyle=':', linewidth=1.5, label='Solved threshold (−110)')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward (moving average)')
ax.set_title('MountainCar-v0: Best Result Per Algorithm (first 10K eps)')
ax.legend(loc='upper right')
ax.set_ylim(-210, -80)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save('mountaincar_comparison')

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — DQN: loss explosion comparison (v2 vs v3)
# ════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Standard DQN vs Double DQN
try:
    std_loss = read_csv('results/mountaincar_dqn_20260602_123116/training_log.csv', 'loss')
    ddq_loss = read_csv('results/mountaincar_dqn_20260602_171518/training_log.csv', 'loss')
    std_rew  = read_csv('results/mountaincar_dqn_20260602_123116/training_log.csv', 'reward')
    ddq_rew  = read_csv('results/mountaincar_dqn_20260602_171518/training_log.csv', 'reward')

    ax1.plot(moving_avg(std_loss, 20), color='#e41a1c', label='Standard DQN', linewidth=1.5)
    ax1.plot(moving_avg(ddq_loss, 20), color='#377eb8', label='Double DQN (v3)', linewidth=1.5)
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Huber Loss (20-ep avg)')
    ax1.set_title('DQN: Loss — Standard vs Double')
    ax1.set_yscale('symlog', linthresh=1)
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(moving_avg(std_rew, 50), color='#e41a1c', label='Standard DQN', linewidth=1.5)
    ax2.plot(moving_avg(ddq_rew, 50), color='#377eb8', label='Double DQN (v3)', linewidth=1.5)
    ax2.axhline(y=-110, color='black', linestyle=':', linewidth=1.5, label='Solved (−110)')
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Reward (50-ep avg)')
    ax2.set_title('DQN: Performance — Standard vs Double')
    ax2.legend(); ax2.grid(True, alpha=0.3)
except Exception as e:
    print(f'  DQN comparison: {e}')

plt.suptitle('Q-Value Overestimation: Standard DQN vs Double DQN', fontsize=13, fontweight='bold')
plt.tight_layout()
save('dqn_overestimation')

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — PPO: goal density comparison v2 vs v5
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

def read_ppo_avg(logfile):
    import re
    try:
        text = open(logfile).read()
        entries = re.findall(r'Episode\s+(\d+) \| Avg raw\(100\): ([^\|]+)\|', text)
        return [(int(e), float(a.strip())) for e, a in entries]
    except:
        return []

v2 = read_ppo_avg('mc_ppo_v2_run.log')
v5 = read_ppo_avg('mc_ppo_v5_run.log')

if v2:
    eps2, avgs2 = zip(*v2)
    axes[0].plot(eps2, avgs2, color='#e41a1c', linewidth=1.5, label='v2 (fixed LR, no fixes)')
if v5:
    eps5, avgs5 = zip(*v5)
    axes[0].plot(eps5, avgs5, color='#377eb8', linewidth=1.5, label='v5 (LR floor, GAE, fixes)')
axes[0].axhline(y=-110, color='black', linestyle=':', linewidth=1.5, label='Solved (−110)')
axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Mean reward (last 100)')
axes[0].set_title('PPO: v2 vs v5 Average Reward')
axes[0].set_ylim(-210, -80); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Goal density bar chart
import re

def count_goals_by_window(logfile, window=1000, max_ep=9000):
    try:
        text = open(logfile).read()
        goals = [int(g) for g in re.findall(r'Episode\s+(\d+) REACHED GOAL', text)]
        windows = range(0, max_ep, window)
        return [sum(1 for g in goals if s <= g < s+window) for s in windows], list(windows)
    except:
        return [], []

counts_v2, wins_v2 = count_goals_by_window('mc_ppo_v2_run.log')
counts_v5, wins_v5 = count_goals_by_window('mc_ppo_v5_run.log')

x = np.arange(len(wins_v5))
w = 0.35
if counts_v2 and counts_v5:
    axes[1].bar(x - w/2, counts_v2[:len(x)], w, label='v2 (no fixes)', color='#e41a1c', alpha=0.8)
    axes[1].bar(x + w/2, counts_v5, w, label='v5 (all fixes)', color='#377eb8', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'{s//1000}K' for s in wins_v5], rotation=45)
    axes[1].set_xlabel('Episode window (×1000)'); axes[1].set_ylabel('Goals reached')
    axes[1].set_title('PPO: Goal Discovery by Episode Window')
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')

plt.suptitle('PPO MountainCar: Effect of Implementation Fixes', fontsize=13, fontweight='bold')
plt.tight_layout()
save('ppo_comparison')

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — A2C: actor loss showing zero-gradient fixed point
# ════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

try:
    # run 4 (best AC before bug fix) vs v3 (after all fixes)
    ac_r4 = read_csv('results/mountaincar_ac_20260602_125900/training_log.csv', 'raw_reward')
    ac_v3 = read_csv('results/mountaincar_ac_20260602_171518/training_log.csv', 'raw_reward')

    ax1.plot(moving_avg(ac_r4, 100), color='#e41a1c', linewidth=1.5, label='Run 4 (entropy=0.01, MC returns)')
    ax1.plot(moving_avg(ac_v3, 100), color='#377eb8', linewidth=1.5, label='v3 (GAE + policy gradient fix)')
    ax1.axhline(y=-110, color='black', linestyle=':', linewidth=1.5)
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Reward (100-ep avg)')
    ax1.set_title('A2C: Best vs Fixed Implementation')
    ax1.legend(); ax1.grid(True, alpha=0.3)
except Exception as e:
    print(f'  A2C reward plot: {e}')

try:
    actor_loss_v3 = read_csv('results/mountaincar_ac_20260602_171518/training_log.csv', 'actor_loss')
    critic_loss_v3 = read_csv('results/mountaincar_ac_20260602_171518/training_log.csv', 'critic_loss')
    eps = range(len(actor_loss_v3))

    ax2.plot(eps, np.abs(actor_loss_v3), color='#ff7f00', linewidth=1, alpha=0.7, label='|Actor loss|')
    ax2.plot(eps, critic_loss_v3, color='#377eb8', linewidth=1.5, label='Critic loss')
    ax2.set_yscale('log')
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('A2C v3: Actor Loss → 0 (Zero-Advantage Fixed Point)')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 5000)
except Exception as e:
    print(f'  A2C loss plot: {e}')

plt.suptitle('A2C: Structural Failure on MountainCar', fontsize=13, fontweight='bold')
plt.tight_layout()
save('ac_structural_failure')

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — PPO: entropy collapse (v3) vs healthy (v5)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

def read_ppo_update_log(path, col):
    try:
        return read_csv(path, col)
    except:
        return []

# v2 update log (closest to v3 in terms of structure)
for ax, logpath, label, color in [
    (axes[0], 'results/mountaincar_ppo_20260602_141540/update_log.csv', 'v3 (LR→0, entropy collapsed)', '#e41a1c'),
    (axes[1], 'results/mountaincar_ppo_20260602_195214/update_log.csv',  'v5 (LR floor, stable)',        '#377eb8'),
]:
    try:
        ent = read_csv(logpath, 'entropy')
        ax.plot(ent, color=color, linewidth=1.5)
        ax.axhline(y=np.log(3), color='gray', linestyle='--', linewidth=1, label=f'Uniform π entropy ({np.log(3):.2f})')
        ax.set_xlabel('Policy update'); ax.set_ylabel('Entropy H(π)')
        ax.set_title(f'PPO Entropy: {label}')
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.2)
    except Exception as e:
        print(f'  entropy {label}: {e}')

plt.suptitle('PPO Entropy: LR→0 Collapse vs LR Floor Stability', fontsize=13, fontweight='bold')
plt.tight_layout()
save('ppo_entropy_comparison')

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — CartPole: sample efficiency bar chart
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))

algos = ['Q-Learning', 'DQN', 'A2C', 'PPO']
solved_at = [9798, 225, 1175, 195]
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00']

bars = ax.bar(algos, solved_at, color=colors, alpha=0.85, edgecolor='white', linewidth=1.2)
ax.set_ylabel('Episodes to solve')
ax.set_title('CartPole-v1: Sample Efficiency Comparison')
ax.set_yscale('log')

for bar, val in zip(bars, solved_at):
    ax.text(bar.get_x() + bar.get_width()/2, val * 1.1, str(val),
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
save('cartpole_sample_efficiency')

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — MountainCar trajectory plot (PPO v5 best policy)
# ════════════════════════════════════════════════════════════════════════════
import shutil
src = 'results/mountaincar_ppo_20260602_195214/trajectory.png'
dst = 'latex/figures/ppo_trajectory.png'
if os.path.exists(src):
    shutil.copy(src, dst)
    print(f'Copied {dst}')

# Also copy the DQN v3 trajectory
src2 = 'results/mountaincar_dqn_20260602_171518/trajectory.png'
dst2 = 'latex/figures/dqn_trajectory.png'
if os.path.exists(src2):
    shutil.copy(src2, dst2)
    print(f'Copied {dst2}')

# Copy training results pngs for the final best runs
for name, src in [
    ('ppo_v5_training',   'results/mountaincar_ppo_20260602_195214/training_results.png'),
    ('dqn_v3_training',   'results/mountaincar_dqn_20260602_171518/training_results.png'),
    ('ac_v3_training',    'results/mountaincar_ac_20260602_171518/training_results.png'),
    ('ql_training',       'results/mountaincar_qlearning_20260602_110124/training_results.png'),
    ('cp_ppo_training',   'results/cartpole_ppo_20260602_103540/training_results.png'),
    ('cp_dqn_training',   'results/cartpole_dqn_20260602_110126/training_results.png'),
    ('cp_ac_training',    'results/cartpole_ac_20260602_110127/training_results.png'),
    ('cp_ql_training',    'results/cartpole_qlearning_20260602_103536/training_results.png'),
    ('gw_training',       'results/gridworld_qlearning_20260602_103536/training_results.png'),
]:
    dst = f'latex/figures/{name}.png'
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f'Copied {dst}')

print('\nAll figures generated in latex/figures/')
print(f'Total files: {len(os.listdir("latex/figures"))}')
