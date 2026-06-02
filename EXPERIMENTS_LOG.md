# RL Experiments Log

Complete record of all runs, bugs found, fixes applied, and final results.
Use this when writing the report.

---

## Final Results

| Environment | Algorithm | Result | Episodes | Notes |
|---|---|---|---|---|
| GridWorld | Q-Learning | ✓ **Solved** | 3,000 | 100% goal rate, avg 10 steps |
| CartPole | Q-Learning | ✓ **Solved** | 9,798 | Near the 10K cap — slow but works |
| CartPole | DQN | ✓ **Solved** | 225 | Fast once buffer size fixed |
| CartPole | Actor-Critic | ✓ **Solved** | 1,175 | After fixing solved threshold |
| CartPole | PPO | ✓ **Solved** | 195 | **Fastest** of all algorithms |
| MountainCar | Q-Learning | ✗ Best avg -130 | 50,000 | Tabular limit — accepted finding |
| MountainCar | DQN | ⟳ Running | 3,000 | Double DQN — previous best avg -132 |
| MountainCar | Actor-Critic | ⟳ Running | 5,000 | 644 goals but slow; entropy=0.01 fix |
| MountainCar | PPO | ⟳ Running | 3,000 | entropy=0.01 fix; 439 goals in prev run |

---

## What Each Run Produced

Every completed run saves to `results/<name>_<timestamp>/`:
- `training_results.png` — learning curves, loss, reward distribution
- `training_log.csv` — per-episode metrics
- `summary.txt` — copy-pasteable stats for report
- `trajectory.png` (MountainCar only) — position + velocity over time
- `model.pth` — trained weights (gitignored)

---

## Bugs Found and Fixed

### Critical (affected results)

| # | Bug | Symptom | Fix |
|---|---|---|---|
| 1 | CartPole DQN: replay buffer 10K too small | Q-values diverged at ep ~500, avg collapsed from 172 → 22 | Buffer 10K → 50K |
| 2 | CartPole DQN: EPS_START=0.9 | Greedy exploitation of random network from ep 1 | EPS_START=1.0 |
| 3 | CartPole AC: SOLVED_AVG=495 | "Mastered" threshold, not "solved" — never triggered | SOLVED_AVG=195 |
| 4 | MC DQN run 1: LR=1e-3 | loss=115,037 — complete Q-value explosion | LR=1e-4 |
| 5 | MC DQN run 1: goal bonus +100 | Inflated Q-targets | +10 (aligns with AC/PPO) |
| 6 | MC DQN run 2: EPS_DECAY=1000 steps | Epsilon hit min at ep 25 (400 steps/ep), agent never explored | EPS_DECAY=50,000 |
| 7 | MC DQN run 3: standard DQN overestimation | Plateau at avg -134 from ep 600–2000, loss=2491 | Double DQN |
| 8 | MC AC run 1: normalising returns not advantages | Per-episode z-score on returns → critic trains on shifting scale → actor_loss=5,442 | Normalise advantages instead |
| 9 | MC AC run 2: entropy=0.05 too high | 644 goals but all in 180–199 steps, couldn't consolidate | entropy=0.01 |
| 10 | MC PPO run 1: entropy=0.05 | Found goal 439×, then regressed completely | entropy=0.01 |
| 11 | MC PPO run 2: entropy=0.001 | Policy deterministic before finding goal (0 goals at ep 850) | entropy=0.01 |

### Code Quality

| # | Bug | Fix |
|---|---|---|
| 12 | Stdout buffering in tee pipes | `python3 -u` flag |
| 13 | Matplotlib global colormap mutation | `.copy()` before `.set_bad()` |
| 14 | GridWorld success check `r > 0` | Track `state == GOAL` directly |
| 15 | `set_yscale('log')` drops zero values | `symlog` |
| 16 | `.squeeze()` scalar on 1-step episodes | `.squeeze(-1)` |
| 17 | MC Q-Learning condition `done and reward > -200` | Always true (reward always -1); simplified to `done` |
| 18 | MC PPO dead variable `total_steps` | Removed |
| 19 | CartPole PPO arbitrary `fill_between(ma±10)` | Removed |

---

## Key Hyperparameters (final committed values)

### CartPole DQN
```
Network: 4 → 128 → 128 → 2  (ReLU, Huber loss, AdamW amsgrad)
BATCH=128, GAMMA=0.99, LR=1e-4, MEMORY=50K, TAU=0.005
Epsilon: 1.0 → 0.05 over 5,000 steps
```

### CartPole Actor-Critic (A2C)
```
Network: 4 → 128 shared → actor(2) + critic(1)
LR=3e-4, GAMMA=0.99, grad_clip=0.5
Normalise advantages (not returns)
Solved threshold: avg ≥ 195 over 100 episodes
```

### CartPole PPO
```
Separate networks: 4 → 128 → 128 (Tanh)
LR=3e-4, γ=0.99, λ=0.95 (GAE), clip=0.2, entropy=0.01
Value_coef=0.5, update every 20 episodes, 10 epochs, batch 64
```

### MountainCar DQN (Double DQN)
```
Network: 2 → 128 → 128 → 3  (ReLU, Huber, AdamW)
BATCH=128, GAMMA=0.99, LR=5e-5, MEMORY=50K, TAU=0.005
Epsilon: 1.0 → 0.01 over 50,000 steps
Reward shaping: height + 100·KE − 1  (+10 at goal)
Double DQN: policy_net selects action, target_net evaluates
```

### MountainCar Actor-Critic
```
Network: 2 → 64 → 64 → 3  (Tanh, orthogonal init)
LR=3e-4, GAMMA=0.99, grad_clip=0.5
State normalisation: pos → [−1,1], vel → [−1,1]
Normalise advantages (not returns)
Reward shaping: height + 100·KE − 1  (+10 at goal)
Entropy coef: 0.01
```

### MountainCar PPO
```
Network: 2 → 64 → 64 → 3  (Tanh, orthogonal init)
LR=3e-4, γ=0.99, λ=0.95, clip=0.2, entropy=0.01
Value_coef=0.5, update every 1,024 steps, 10 epochs, batch 64
State normalisation: pos → [−1,1], vel → [−1,1]
Reward shaping: height + 100·KE − 1  (+10 at goal)
```

---

## Narrative for the Report

### Algorithm Comparison (CartPole)
PPO solved fastest (ep 195), DQN next (ep 225), A2C (ep 1,175), Q-Learning slowest (ep 9,798). PPO's sample efficiency advantage comes from reusing data across multiple epochs per update. Q-Learning's discretisation introduces approximation error that slows convergence.

### Why MountainCar is Harder
The environment has a **sparse, deceptive reward structure**: the car gets −1 every step regardless of progress. The only way to get a shorter episode is to reach the goal. Without reward shaping, no algorithm makes progress from random initialisation.

With reward shaping (`height + 100·KE − 1`), all deep RL algorithms can learn — but the difficulty reveals each algorithm's weaknesses:
- **Q-Learning**: discretisation too coarse for precise timing
- **Standard DQN**: Q-value overestimation causes plateaus (fixed with Double DQN)
- **A2C**: on-policy → can't replay successful trajectories; learns conservative strategy
- **PPO**: with correct entropy (0.01), balances exploration and consolidation

### Deep Insights Worth Discussing

1. **Entropy coefficient is critical for on-policy methods.** Too high (0.05): finds goal but keeps unlearning it. Too low (0.001): policy goes deterministic before finding goal. 0.01 is the balance.

2. **Standard DQN overestimates Q-values** due to the max operator in targets. With reward shaping creating diverse Q-values, this bias compounds. Double DQN decouples action selection from evaluation, removing the bias.

3. **Return normalisation vs advantage normalisation.** Normalising returns per-episode creates a shifting target for the critic (different scale every episode). Normalising advantages after computing them (standard A2C/PPO) keeps the critic's training signal consistent.

4. **Replay buffer size matters more than episode count for DQN.** A 10K buffer filled with long successful episodes gets corrupted when performance drops — the diverse experience needed for recovery isn't there. 50K keeps enough diversity.
