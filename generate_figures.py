"""
Professional paper figures for the RL algorithms comparison report.
Run from /home/taurovinol/rl/.
All figures sized for A4 paper (text width = 5.9 in) at 200 DPI.
"""
import os, csv, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

os.makedirs('latex/figures', exist_ok=True)

# ─── Paper geometry ────────────────────────────────────────────────────────────
# A4 (21 cm) with 3 cm left/right margins → text width = 15 cm = 5.906 in
# Figures embedded at \linewidth; figsize=(FW, H) gives exact 10 pt fonts
FW  = 5.9    # figure width in inches
DPI = 200

# ─── Academic style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':         10,
    'axes.titlesize':    10,
    'axes.titleweight':  'bold',
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   8.5,
    'legend.framealpha': 0.92,
    'legend.edgecolor':  '0.8',
    'legend.borderpad':  0.4,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.22,
    'grid.linestyle':    ':',
    'lines.linewidth':   1.8,
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'savefig.facecolor': 'white',
})

C  = {
    'ql':  '#d62728',   # red
    'dqn': '#1f77b4',   # blue
    'a2c': '#2ca02c',   # green
    'ppo': '#ff7f0e',   # orange
    'v2':  '#9467bd',   # purple (PPO v2, for comparison)
    'ref': '#333333',   # dark gray reference lines
}
LB = {'ql': 'Q-Learning', 'dqn': 'Double DQN', 'a2c': 'A2C', 'ppo': 'PPO (v5)', 'v2': 'PPO (v2)'}

# ─── Helpers ───────────────────────────────────────────────────────────────────
def sma(data, w):
    return np.convolve(data, np.ones(w) / w, mode='valid')

def col(path, name):
    with open(path) as f:
        return np.array([float(r[name]) for r in csv.DictReader(f)])

def plabel(ax, letter):
    ax.text(0.02, 0.97, f'({letter})', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')

def savefig(name):
    plt.savefig(f'latex/figures/{name}.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  {name}.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1  CartPole — all four algorithms
# ═══════════════════════════════════════════════════════════════════════════════
print('Figure 1 — CartPole overview')
fig = plt.figure(figsize=(FW, 3.5))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1.65, 1.0], wspace=0.38, figure=fig)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

runs_cp = [
    ('ql',  'results/cartpole_qlearning_20260602_103536/training_log.csv', 'reward', 100, 9798),
    ('dqn', 'results/cartpole_dqn_20260602_110126/training_log.csv',       'reward',  20,  225),
    ('a2c', 'results/cartpole_ac_20260602_110127/training_log.csv',        'reward',  50, 1175),
    ('ppo', 'results/cartpole_ppo_20260602_103540/training_log.csv',       'reward',  10,  195),
]
for key, path, cn, w, ep in runs_cp:
    r  = col(path, cn)
    mv = sma(r, w)
    ax1.plot(range(w - 1, len(r)), mv, color=C[key],
             label=f'{LB[key]} (ep. {ep:,})', lw=1.8)
    ax1.axvline(ep, color=C[key], ls='--', lw=0.9, alpha=0.5)

ax1.axhline(195, color=C['ref'], ls=':', lw=1.2, label='Solved (195)')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward (moving avg.)')
ax1.set_xlim(-100, 10300)
ax1.set_ylim(-5, 510)
ax1.legend(loc='upper left', fontsize=8)
plabel(ax1, 'a')

names  = ['Q-Learning', 'A2C', 'DQN', 'PPO']
values = [9798, 1175, 225, 195]
colors = [C['ql'], C['a2c'], C['dqn'], C['ppo']]
bars = ax2.barh(names, values, color=colors, height=0.55, edgecolor='none')
ax2.set_xscale('log')
ax2.set_xlabel('Episodes to solve')
ax2.spines['left'].set_visible(False)
ax2.tick_params(left=False)
for bar, v in zip(bars, values):
    ax2.text(v * 1.2, bar.get_y() + bar.get_height() / 2,
             f'{v:,}', va='center', fontsize=8.5, color='#333')
ax2.set_xlim(80, 60000)
ax2.grid(False)
ax2.grid(axis='x', alpha=0.22, linestyle=':')
plabel(ax2, 'b')

plt.suptitle('CartPole-v1: All Four Algorithms', y=1.01, fontsize=11, fontweight='bold')
savefig('fig_cartpole')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2  GridWorld — training curves
# ═══════════════════════════════════════════════════════════════════════════════
print('Figure 2 — GridWorld training curves')
fig, axes = plt.subplots(1, 2, figsize=(FW, 2.8))
GW_PATH = 'results/gridworld_qlearning_20260602_103536/training_log.csv'
gw_r  = col(GW_PATH, 'reward')
gw_s  = col(GW_PATH, 'steps')
eps_gw = np.arange(1, len(gw_r) + 1)
W = 50

ax = axes[0]
ax.plot(eps_gw, gw_r, alpha=0.2, color=C['ql'], lw=0.5)
ax.plot(range(W, len(gw_r) + 1), sma(gw_r, W), color=C['ql'], lw=2.0, label=f'{W}-ep avg')
ax.axhline(9, color='#2ca02c', ls='--', lw=1.2, label='Near-optimal (9)')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.legend(fontsize=8.5)
plabel(ax, 'a')

ax = axes[1]
ax.plot(eps_gw, gw_s, alpha=0.2, color='#ff7f0e', lw=0.5)
ax.plot(range(W, len(gw_s) + 1), sma(gw_s, W), color='#ff7f0e', lw=2.0)
ax.axhline(10, color='#2ca02c', ls='--', lw=1.2, label='Optimal (10 steps)')
ax.set_xlabel('Episode')
ax.set_ylabel('Steps to goal')
ax.legend(fontsize=8.5)
plabel(ax, 'b')

plt.suptitle('GridWorld Q-Learning: Training Progress', y=1.01, fontsize=11, fontweight='bold')
plt.tight_layout()
savefig('fig_gridworld_training')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3  GridWorld — learned value function and greedy policy
# ═══════════════════════════════════════════════════════════════════════════════
print('Figure 3 — GridWorld policy')
GRID_H, GRID_W = 6, 6
WALLS  = {(1,1), (1,3), (2,1), (3,3), (3,4), (4,2)}
START, GOAL = (0, 0), (5, 5)
ARROW_CH = ['↑', '→', '↓', '←']

q_table = np.load('results/gridworld_qlearning_20260602_103536/q_table.npy')

fig, axes = plt.subplots(1, 2, figsize=(FW, 2.8))

# (a) Value function
ax = axes[0]
vmap = np.max(q_table, axis=2).astype(float)
for r, c in WALLS:
    vmap[r, c] = np.nan
cmap = plt.cm.RdYlGn.copy()
cmap.set_bad('#2a2a2a')
im = ax.imshow(vmap, cmap=cmap, interpolation='nearest', aspect='equal')
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('V(s)', fontsize=9)
cbar.ax.tick_params(labelsize=8)
ax.set_xticks(range(GRID_W))
ax.set_yticks(range(GRID_H))
ax.tick_params(length=0, labelsize=8)
ax.set_xlabel('Column', fontsize=9)
ax.set_ylabel('Row', fontsize=9)
ax.text(START[1], START[0], 'S', ha='center', va='center',
        fontsize=11, fontweight='bold', color='#1565C0')
ax.text(GOAL[1], GOAL[0], 'G', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white')
ax.grid(False)
ax.spines[:].set_visible(False)
plabel(ax, 'a')

# (b) Greedy policy
ax = axes[1]
ax.set_xlim(-0.5, GRID_W - 0.5)
ax.set_ylim(GRID_H - 0.5, -0.5)
ax.set_xticks(range(GRID_W))
ax.set_yticks(range(GRID_H))
ax.tick_params(length=0, labelsize=8)
ax.set_xlabel('Column', fontsize=9)
ax.set_ylabel('Row', fontsize=9)
ax.grid(True, alpha=0.35, lw=0.6, color='#ccc')
ax.set_axisbelow(True)
ax.spines[:].set_visible(False)

for r in range(GRID_H):
    for c in range(GRID_W):
        if (r, c) in WALLS:
            ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                             fc='#2a2a2a', ec='none', zorder=2))
        elif (r, c) == GOAL:
            ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                             fc='#C8E6C9', ec='none', zorder=2))
            ax.text(c, r, 'G', ha='center', va='center',
                    fontsize=11, fontweight='bold', color='#2ca02c', zorder=3)
        elif (r, c) == START:
            ax.text(c, r, ARROW_CH[np.argmax(q_table[r, c])], ha='center', va='center',
                    fontsize=14, color='#1565C0', fontweight='bold', zorder=3)
        else:
            ax.text(c, r, ARROW_CH[np.argmax(q_table[r, c])], ha='center', va='center',
                    fontsize=14, color='#333333', zorder=3)

plabel(ax, 'b')
plt.suptitle('GridWorld: Learned Value Function and Greedy Policy',
             y=1.01, fontsize=11, fontweight='bold')
plt.tight_layout()
savefig('fig_gridworld_policy')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4  MountainCar — algorithm comparison
# ═══════════════════════════════════════════════════════════════════════════════
print('Figure 4 — MountainCar comparison')
fig, ax = plt.subplots(figsize=(FW, 3.5))

mc_cfgs = [
    ('ql',  'results/mountaincar_qlearning_20260602_110124/training_log.csv', 'reward',     500),
    ('dqn', 'results/mountaincar_dqn_20260602_171518/training_log.csv',       'reward',     100),
    ('ppo', 'results/mountaincar_ppo_20260602_195214/training_log.csv',       'raw_reward', 100),
]
for key, path, cn, w in mc_cfgs:
    r    = col(path, cn)
    mv   = sma(r, w)
    n    = min(len(r), 10000)
    xpts = list(range(w - 1, n))
    ax.plot(xpts, mv[:len(xpts)], color=C[key], label=LB[key], lw=1.8)

ax.axhline(-110, color=C['ref'], ls=':', lw=1.2, label='Solved threshold (−110)')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward (moving avg.)')
ax.set_ylim(-212, -80)
ax.set_xlim(0, 10300)
ax.legend(loc='upper right')
ax.set_title('MountainCar-v0: Best Result per Algorithm (first 10 K episodes)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
savefig('fig_mountaincar')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5  DQN — Q-value overestimation
# ═══════════════════════════════════════════════════════════════════════════════
print('Figure 5 — DQN overestimation')
fig, axes = plt.subplots(1, 2, figsize=(FW, 3.0))

std_l = col('results/mountaincar_dqn_20260602_123116/training_log.csv', 'loss')
ddq_l = col('results/mountaincar_dqn_20260602_171518/training_log.csv', 'loss')
std_r = col('results/mountaincar_dqn_20260602_123116/training_log.csv', 'reward')
ddq_r = col('results/mountaincar_dqn_20260602_171518/training_log.csv', 'reward')

ax = axes[0]
w = 20
ax.plot(range(w-1, len(std_l)), sma(std_l, w), color='#d62728', label='Standard DQN', lw=1.8)
ax.plot(range(w-1, len(ddq_l)), sma(ddq_l, w), color=C['dqn'],  label='Double DQN',   lw=1.8)
ax.set_yscale('symlog', linthresh=1)
ax.set_xlabel('Episode')
ax.set_ylabel('Huber loss (symlog scale)')
ax.legend(fontsize=8.5)
plabel(ax, 'a')

ax = axes[1]
w = 50
ax.plot(range(w-1, len(std_r)), sma(std_r, w), color='#d62728', label='Standard DQN', lw=1.8)
ax.plot(range(w-1, len(ddq_r)), sma(ddq_r, w), color=C['dqn'],  label='Double DQN',   lw=1.8)
ax.axhline(-110, color=C['ref'], ls=':', lw=1.2, label='Solved (−110)')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward (moving avg.)')
ax.set_ylim(-212, -90)
ax.legend(fontsize=8.5)
plabel(ax, 'b')

plt.suptitle('Q-Value Overestimation: Standard DQN vs Double DQN',
             y=1.01, fontsize=11, fontweight='bold')
plt.tight_layout()
savefig('fig_dqn_overestimation')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6  A2C — zero-advantage fixed point
# ═══════════════════════════════════════════════════════════════════════════════
print('Figure 6 — A2C structural failure')
fig, axes = plt.subplots(1, 2, figsize=(FW, 3.0))

ac_r  = col('results/mountaincar_ac_20260602_171518/training_log.csv', 'raw_reward')
ac_al = col('results/mountaincar_ac_20260602_171518/training_log.csv', 'actor_loss')
ac_cl = col('results/mountaincar_ac_20260602_171518/training_log.csv', 'critic_loss')
eps_ac = np.arange(1, len(ac_r) + 1)

ax = axes[0]
w = 100
ax.plot(eps_ac, ac_r, alpha=0.18, color=C['a2c'], lw=0.5)
ax.plot(range(w, len(ac_r) + 1), sma(ac_r, w), color=C['a2c'], lw=2.0, label=f'{w}-ep avg')
ax.axhline(-200, color='#aaa', ls=':', lw=1.0)
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.set_ylim(-205, -170)
ax.legend(fontsize=8.5)
plabel(ax, 'a')

ax = axes[1]
ax.plot(eps_ac, np.abs(ac_al), color='#ff7f0e', lw=1.0, alpha=0.65, label='|Actor loss|')
ax.plot(eps_ac, ac_cl,         color=C['dqn'],  lw=1.6,              label='Critic loss')
ax.set_yscale('log')
ax.axvline(600, color='#aaa', ls='--', lw=1.0)
ax.text(650, ax.get_ylim()[0] * 5, 'ep. 600\nactor→0', fontsize=7.5, color='#888', va='bottom')
ax.set_xlabel('Episode')
ax.set_ylabel('Loss (log scale)')
ax.legend(fontsize=8.5)
plabel(ax, 'b')

plt.suptitle('A2C: Zero-Advantage Fixed Point on MountainCar',
             y=1.01, fontsize=11, fontweight='bold')
plt.tight_layout()
savefig('fig_ac_failure')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7  PPO — v2 regression vs v5 stable, with goal counts
# ═══════════════════════════════════════════════════════════════════════════════
print('Figure 7 — PPO improvements')
fig = plt.figure(figsize=(FW, 3.2))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1.0], wspace=0.40, figure=fig)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

ppo_v2 = col('results/mountaincar_ppo_20260602_171518/training_log.csv', 'raw_reward')
ppo_v5 = col('results/mountaincar_ppo_20260602_195214/training_log.csv', 'raw_reward')

w = 100
mv2 = sma(ppo_v2, w);  x2 = list(range(w - 1, len(ppo_v2)))
mv5 = sma(ppo_v5, w);  x5 = list(range(w - 1, len(ppo_v5)))
ax1.plot(x2, mv2, color=C['v2'],  lw=1.8, label='v2 (fixed LR)')
ax1.plot(x5, mv5, color=C['ppo'], lw=1.8, label='v5 (LR floor)')
ax1.axhline(-110, color=C['ref'], ls=':', lw=1.2, label='Solved (−110)')
ax1.axvline(5300, color=C['v2'], ls='--', lw=0.9, alpha=0.55)
ax1.text(5350, -118, 'collapse', fontsize=7.5, color=C['v2'], va='top')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward (moving avg.)')
ax1.set_ylim(-212, -88)
ax1.legend(fontsize=8)
plabel(ax1, 'a')

def goals_per_window(rewards, window=1000):
    return [int(np.sum(np.array(rewards[s:s+window]) > -199.5))
            for s in range(0, len(rewards), window)]

g2 = goals_per_window(ppo_v2)
g5 = goals_per_window(ppo_v5)
n  = max(len(g2), len(g5))
g2 += [0] * (n - len(g2))
g5 += [0] * (n - len(g5))
x  = np.arange(n)
bw = 0.38
ax2.bar(x - bw/2, g2, bw, color=C['v2'],  label='v2', edgecolor='none', alpha=0.85)
ax2.bar(x + bw/2, g5, bw, color=C['ppo'], label='v5', edgecolor='none', alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels([f'{i+1}K' for i in range(n)], rotation=45, fontsize=7.5, ha='right')
ax2.set_xlabel('Episode window (×1,000)')
ax2.set_ylabel('Goals reached')
ax2.legend(fontsize=8)
plabel(ax2, 'b')

plt.suptitle('PPO: Effect of Learning Rate Schedule Fix (v2 vs v5)',
             y=1.01, fontsize=11, fontweight='bold')
savefig('fig_ppo_improvements')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8  PPO — entropy collapse vs stable
# ═══════════════════════════════════════════════════════════════════════════════
print('Figure 8 — PPO entropy')
fig, axes = plt.subplots(1, 2, figsize=(FW, 2.8))
unif = np.log(3)   # uniform over 3 actions

ent_v3 = col('results/mountaincar_ppo_20260602_141540/update_log.csv',  'entropy')
ent_v5 = col('results/mountaincar_ppo_20260602_195214/update_log.csv',  'entropy')

for ax, ent, title, clr, ltr in [
    (axes[0], ent_v3, 'v3: LR→0 (entropy collapse)', '#d62728', 'a'),
    (axes[1], ent_v5, 'v5: LR floor (stable)',        C['ppo'],  'b'),
]:
    ax.plot(range(1, len(ent)+1), ent, color=clr, lw=1.6, alpha=0.85)
    ax.axhline(unif, color='#999', ls='--', lw=1.0, label=f'Uniform H = {unif:.2f}')
    ax.set_xlabel('Policy update')
    ax.set_ylabel('Entropy H(π)')
    ax.set_ylim(-0.05, 1.30)
    ax.set_title(title)
    ax.legend(fontsize=8)
    plabel(ax, ltr)

plt.suptitle('PPO: Entropy Dynamics Under Different LR Schedules',
             y=1.01, fontsize=11, fontweight='bold')
plt.tight_layout()
savefig('fig_ppo_entropy')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 9  Trajectories — PPO v5 and DQN v3
# ═══════════════════════════════════════════════════════════════════════════════
print('Figure 9 — Trajectories')
p_ppo = 'latex/figures/ppo_trajectory.png'
p_dqn = 'latex/figures/dqn_trajectory.png'
if os.path.exists(p_ppo) and os.path.exists(p_dqn):
    fig, axes = plt.subplots(2, 1, figsize=(FW, 4.0))
    for ax, path, title in [
        (axes[0], p_ppo, '(a)  PPO v5 — test episode trajectory'),
        (axes[1], p_dqn, '(b)  Double DQN v3 — test episode trajectory'),
    ]:
        ax.imshow(mpimg.imread(path), aspect='auto')
        ax.axis('off')
        ax.set_title(title, fontsize=9, loc='left', pad=3)
    plt.suptitle('MountainCar-v0: Learned Trajectory (Position and Velocity)',
                 y=1.01, fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig('fig_trajectories')
else:
    print('  trajectory PNGs not found — skipping')


print('\nAll figures saved to latex/figures/')
print(f'Total: {len([f for f in os.listdir("latex/figures") if f.endswith(".png")])} PNG files')
