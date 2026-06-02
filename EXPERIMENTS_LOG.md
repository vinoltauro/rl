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
| MountainCar | Q-Learning | ⟳ **Running v2** | 50,000 | Raw rewards (shaping caused regression — see bug #28) |
| MountainCar | DQN | ⟳ **Running v2** | 5,000 | MAX_STEPS 400→200, N_EPISODES 3000→5000 |
| MountainCar | Actor-Critic | ⟳ **Running v2** | 5,000 | GAE(λ=0.95) + exact entropy + truncation fix |
| MountainCar | PPO | ⟳ **Running v2** | 8,000 | N_EPISODES 3000→8000 + truncation fix |

All 4 MountainCar v2 runs started **2026-06-02 14:15** in `rl_runs` tmux session (windows: mc_ql_v2, mc_dqn_v2, mc_ac_v2, mc_ppo_v2). Logs: `mc_*_v2_run.log`.

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

### Round 1 — Critical (affected results)

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

### Round 2 — Critical (found after full audit, 2026-06-02)

| # | Bug | File | Symptom / Root Cause | Fix |
|---|---|---|---|---|
| 20 | MC Q-Learning: no reward shaping | `1_q_learning.py` | Uses raw reward (-1/step). No gradient signal toward goal between successful eps. DQN/AC/PPO all used shaped rewards — unfair comparison, Q-table starved of information | Added `shape_reward(height + 100·KE − 1)`. Q-table updated with shaped reward; raw reward tracked separately for solved check |
| 21 | MC DQN: MAX_STEPS=400 vs 200 everywhere else | `2_dqn.py` | Wrong episode length creates incomparable results (Avg -400 at timeout vs -200), halves buffer diversity (125 eps in 50K buffer vs 250), and doubles wall-clock time per run | MAX_STEPS 400 → 200 |
| 22 | MC DQN: N_EPISODES=3000 too few | `2_dqn.py` | With Double DQN plateau at -132, needed more steps to consolidate. Budget was ~600K steps, needs ~1M | N_EPISODES 3000 → 5000 |
| 23 | MC DQN: summary template says "+100 at goal" | `2_dqn.py:222` | Documentation bug — code has +10 but summary text said +100 | Fixed to "+10 at goal" |
| 24 | MC AC: full MC returns instead of GAE | `3_actor_critic.py` | Full Monte Carlo over 150–200 step episodes. Variance of return estimate is enormous (shaped rewards ±1.5/step × 200 steps). Critic never converges reliably. This is the primary reason AC underperformed PPO by 30 steps avg despite same algorithm family | Replaced with GAE(λ=0.95). All_vals list encodes terminal/truncation correctly via bootstrap value |
| 25 | MC AC: entropy = -mean_log_prob (noisy estimator) | `3_actor_critic.py:135` | Used single-sample E[−log π(a\|s)] instead of exact H(π). Higher variance per update; PPO used `dist.entropy()` (exact) — inconsistency between implementations | Changed to `dist.entropy().mean()` at each step, accumulated and averaged |
| 26 | MC AC + PPO: truncation bias in GAE/returns | `3_actor_critic.py`, `4_ppo.py` | `done = terminated or truncated`. Bootstrap zeroed for BOTH true terminals AND timeouts. Timeout episodes should bootstrap V(s_final) ≠ 0, since the episode continues from that state in principle. With >50% timeout episodes, this systematically underestimates V(s), biasing advantages | AC: explicit bootstrap from V(final_state) when truncated. PPO: store `terminated` separately, use it (not `done`) in GAE bootstrap mask |
| 27 | MC PPO: N_EPISODES=3000 insufficient | `4_ppo.py` | PPO was finding goal (avg -162) but couldn't consolidate to -110 within budget. ~480K steps needed ~1M+ for MountainCar-v0 | N_EPISODES 3000 → 8000 (~1.3M steps) |
| 28 | MC Q-Learning: reward shaping caused regression | `1_q_learning.py` | v2 run with shaped reward stayed at avg -200 through all 44,000 episodes (vs -130 with raw rewards in v1). Shaped reward `height + 100·KE - 1` creates a locally optimal greedy policy that oscillates in a high-momentum region without crossing pos=0.5. With ε=0.001 since ep 11K, Q-table has no exploration to escape this attractor. Deep RL escapes via function approximation noise; tabular with a converged greedy policy cannot. This is an empirical demonstration of Ng et al. (1999): non-potential-based shaping can alter the optimal policy. | Reverted to raw rewards. Kept shape_reward() in code with full explanation. Tabular Q-Learning uses raw rewards — this is the correct valid comparison. |

### Round 1 — Code Quality

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

## Key Hyperparameters (v2 — final committed values)

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

### MountainCar Q-Learning (v2)
```
Tabular: 40×40 bins over pos×vel = 1,600 states
LR=0.3, GAMMA=0.995
Epsilon: 1.0 → 0.001, decay 0.9997 (hits min at ~ep 11K)
N_EPISODES=50,000, MAX_STEPS=200
Reward shaping: height + 100·KE − 1  (+10 at goal)   ← NEW in v2
Q-table updated with shaped reward; raw reward for solve check
```

### MountainCar DQN (v2 — Double DQN)
```
Network: 2 → 128 → 128 → 3  (ReLU, Huber, AdamW)
BATCH=128, GAMMA=0.99, LR=5e-5, MEMORY=50K, TAU=0.005
Epsilon: 1.0 → 0.01 over 50,000 steps
N_EPISODES=5,000, MAX_STEPS=200   ← both changed in v2
Reward shaping: height + 100·KE − 1  (+10 at goal)
Double DQN: policy_net selects action, target_net evaluates
```

### MountainCar Actor-Critic (v2)
```
Network: 2 → 64 → 64 → 3  (Tanh, orthogonal init)
LR=3e-4, GAMMA=0.99, LAM=0.95 (GAE), grad_clip=0.5   ← GAE new in v2
N_EPISODES=5,000, MAX_STEPS=200
State normalisation: pos → [−1,1], vel → [−1,1]
Reward shaping: height + 100·KE − 1  (+10 at goal)
Entropy coef: 0.01 via exact dist.entropy()            ← was noisy -log_prob
Truncation fix: V(final_state) bootstrap on timeout    ← new in v2
```

### MountainCar PPO (v2)
```
Network: 2 → 64 → 64 → 3  (Tanh, orthogonal init)
LR=3e-4, γ=0.99, λ=0.95, clip=0.2, entropy=0.01
Value_coef=0.5, update every 1,024 steps, 10 epochs, batch 64
N_EPISODES=8,000, MAX_STEPS=200   ← N_EPISODES tripled in v2
State normalisation: pos → [−1,1], vel → [−1,1]
Reward shaping: height + 100·KE − 1  (+10 at goal)
Truncation fix: terminated vs truncated in GAE bootstrap  ← new in v2
```

---

## MountainCar Run History

### Q-Learning
| Run | Episodes | Last-100 avg | Best avg | Notes |
|---|---|---|---|---|
| run 1 | 25,000 | -138.11 | -123.72 | Raw reward |
| run 2 | 50,000 | -137.82 | -130.05 | Raw reward |
| v2 (killed) | 44,000 | -200.00 | -200.00 | Shaped reward — stuck in local attractor (bug #28) |
| **v2 restart** | 50,000 | ⟳ | ⟳ | **Raw reward — correct approach** |

### DQN
| Run | Episodes | Last-100 avg | Notes |
|---|---|---|---|
| run 1 | 600 | -211.20 | LR=1e-3, loss exploded (115K) |
| run 2 | 1,000 | -132.02 | EPS_DECAY too fast, success 24% |
| run 3 | 2,000 | -134.19 | Standard DQN, overestimation plateau |
| run 4 | 3,000 | -134 (est) | Double DQN introduced; MAX_STEPS=400 |
| run 5 | killed @ep250 | n/a | Killed before useful data; same bugs |
| **v2** | 5,000 | ⟳ | **MAX_STEPS=200, Double DQN** |

### Actor-Critic
| Run | Episodes | Last-100 avg | Notes |
|---|---|---|---|
| run 1 | 2,000 | -200.00 | Returns normalised (not advantages) — actor loss 6,344 |
| run 2 | 2,000 | -200.00 | Advantages normalised; entropy=0.05 still too high |
| run 3 | 3,000 | -200.00 | entropy=0.05 — actor loss 5,442 |
| run 4 | 3,000 | -188.91 | entropy=0.01 — improved but MC variance too high |
| **v2** | 5,000 | ⟳ | **GAE(λ=0.95) + exact entropy + truncation fix** |

### PPO
| Run | Episodes | Last-100 avg | Notes |
|---|---|---|---|
| run 1 | 1,000 | -161.59 | entropy=0.05, 1000 eps |
| run 2 | 3,000 | -200.00 | entropy=0.001 — policy collapsed |
| run 3 | 3,000 | -200.00 | entropy=0.05 — entropy collapse at end |
| run 4 | 3,000 | -162.37 | entropy=0.01 ✓ — right setting, insufficient budget |
| run 5 | 3,000 | -142.70 | entropy=0.01 — same budget, same outcome |
| **v2** | 8,000 | ⟳ | **8,000 episodes + truncation fix** |

---

## Narrative for the Report

### Algorithm Comparison (CartPole)
PPO solved fastest (ep 195), DQN next (ep 225), A2C (ep 1,175), Q-Learning slowest (ep 9,798). PPO's sample efficiency advantage comes from reusing data across multiple epochs per update. Q-Learning's discretisation introduces approximation error that slows convergence.

### Why MountainCar is Harder
The environment has a **sparse, deceptive reward structure**: the car gets −1 every step regardless of progress. The only way to get a shorter episode is to reach the goal. Without reward shaping, no algorithm makes progress from random initialisation.

With reward shaping (`height + 100·KE − 1`), all deep RL algorithms can learn — but the difficulty reveals each algorithm's weaknesses:
- **Q-Learning**: discretisation too coarse for precise timing; also lacked reward shaping in v1 (unfair handicap)
- **Standard DQN**: Q-value overestimation causes plateaus (fixed with Double DQN)
- **A2C v1**: MC returns over 200-step episodes create enormous variance; policy never converged reliably
- **A2C v2**: GAE(λ=0.95) dramatically reduces variance — expected significant improvement
- **PPO**: correct entropy (0.01) + sufficient budget (8K eps) should be enough to solve

### Deep Insights Worth Discussing

1. **Entropy coefficient is critical for on-policy methods.** Too high (0.05): finds goal but keeps unlearning it. Too low (0.001): policy goes deterministic before finding goal. 0.01 is the balance.

2. **Standard DQN overestimates Q-values** due to the max operator in targets. With reward shaping creating diverse Q-values, this bias compounds. Double DQN decouples action selection from evaluation, removing the bias.

3. **Return normalisation vs advantage normalisation.** Normalising returns per-episode creates a shifting target for the critic (different scale every episode). Normalising advantages after computing them (standard A2C/PPO) keeps the critic's training signal consistent.

4. **Replay buffer size matters more than episode count for DQN.** A 10K buffer filled with long successful episodes gets corrupted when performance drops — the diverse experience needed for recovery isn't there. 50K keeps enough diversity.

5. **GAE vs MC returns in A2C.** For MountainCar episodes of 150–200 steps, MC return variance is `Var(∑ γᵗrₜ)`. With shaped rewards that vary ±1.5/step, accumulated variance over 200 steps is dominated by early steps (γ^t discounts later ones less). GAE(λ=0.95) trades small bias for dramatic variance reduction — equivalent to a weighted blend of 1-step through ∞-step returns. The critic can now train on a stable target.

6. **Truncation bias is a systematic underestimation of V(s).** When an episode times out (truncated, not terminated), the environment would continue from the current state if it could. Treating timeout as a true terminal (V=0 bootstrap) tells the critic "there's no value here" — wrong, since the agent could still reach the goal from that state. This biases all advantage estimates toward zero, making gradients weak early in training.

7. **Reward shaping must be consistent across algorithms.** Q-Learning in v1 used raw reward (−1/step) while all deep methods used `height + 100·KE − 1`. This meant Q-Learning had zero information about progress between random goal discoveries, while deep methods had a continuous gradient. The "tabular limit" finding in v1 partly reflected this information handicap, not just discretisation limits.
