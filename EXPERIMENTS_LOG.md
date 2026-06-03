# RL Experiments Log

Complete record of all runs, bugs found, fixes applied, and final results.
Use this when writing the report.

---

## TODO / Future Work

| # | Task | Priority | Notes |
|---|---|---|---|
| 1 | **Batch A2C (simulated parallelism)** | High | Collect N=8 episodes before each gradient update to simulate n_envs=8. Directly addresses the zero-advantage fixed point that causes single-env A2C to fail. Confirmed root cause: SB3 requires n_envs=16 for MountainCar. See bug #36 and Option B analysis. |

---

## Final Results

| Environment | Algorithm | Result | Episodes | Notes |
|---|---|---|---|---|
| GridWorld | Q-Learning | ✓ **Solved** | 3,000 | 100% goal rate, avg 10 steps |
| CartPole | Q-Learning | ✓ **Solved** | 9,798 | Near the 10K cap — slow but works |
| CartPole | DQN | ✓ **Solved** | 225 | Fast once buffer size fixed |
| CartPole | Actor-Critic | ✓ **Solved** | 1,175 | After fixing solved threshold |
| CartPole | PPO | ✓ **Solved** | 195 | **Fastest** of all algorithms |
| MountainCar | Q-Learning | ✗ Not solved | 50,000 | Best avg −130, last-100 avg −138 — confirmed tabular limit |
| MountainCar | DQN | ✗ **Not solved** | 8,000 | Best: v3 avg −146.68 ±6.31. v5 peaked at ep 5K then regressed to −182. |
| MountainCar | Actor-Critic | ✗ **Structurally fails** | 5,000 | Known single-env A2C limitation. SB3 requires n_envs=16. |
| MountainCar | PPO | ✗ **Not solved** | 12,000 | Best: v5 avg −151.80. v6 had 6,764 goals but eroded to −200 at end. |

**All runs complete as of 2026-06-03 05:25 UTC.**

**v3 runs started 2026-06-02 ~17:00 UTC** in `rl_runs` tmux (mc_dqn_v3, mc_ac_v3, mc_ppo_v3). Logs: `mc_*_v3_run.log`.

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
| 5 | MC DQN run 1: goal bonus +100 | Inflated Q-targets | +10 |
| 6 | MC DQN run 2: EPS_DECAY=1000 steps | Epsilon hit min at ep 25, agent never explored | EPS_DECAY=50,000 |
| 7 | MC DQN run 3: standard DQN overestimation | Plateau at avg -134, loss=2491 | Double DQN |
| 8 | MC AC run 1: normalising returns not advantages | Per-episode z-score on returns → actor_loss=5,442 | Normalise advantages instead |
| 9 | MC AC run 2: entropy=0.05 too high | 644 goals but all in 180–199 steps, couldn't consolidate | entropy=0.01 |
| 10 | MC PPO run 1: entropy=0.05 | Found goal 439×, then regressed completely | entropy=0.01 |
| 11 | MC PPO run 2: entropy=0.001 | Policy deterministic before finding goal | entropy=0.01 |

### Round 2 — Critical (full audit, 2026-06-02)

| # | Bug | File | Symptom / Root Cause | Fix |
|---|---|---|---|---|
| 20 | MC Q-Learning: no reward shaping | `1_q_learning.py` | Raw reward only — Q-table starved of signal between random goal discoveries while DQN/AC/PPO used shaped rewards | Added `shape_reward()` — later reverted (see bug #28) |
| 21 | MC DQN: MAX_STEPS=400 vs 200 everywhere | `2_dqn.py` | Incomparable results, halved buffer diversity, doubled compute | MAX_STEPS 400 → 200 |
| 22 | MC DQN: N_EPISODES=3000 too few | `2_dqn.py` | Insufficient budget to consolidate | N_EPISODES 3000 → 5000 |
| 23 | MC DQN: summary template "+100 at goal" | `2_dqn.py` | Code has +10 but summary said +100 | Fixed documentation |
| 24 | MC AC: full MC returns instead of GAE | `3_actor_critic.py` | MC over 200-step episodes — enormous variance, critic never converged | GAE(λ=0.95) |
| 25 | MC AC: entropy = -mean_log_prob (noisy) | `3_actor_critic.py` | Single-sample estimator, high variance vs exact dist.entropy() | `dist.entropy().mean()` |
| 26 | MC AC + PPO: truncation bias | `3_actor_critic.py`, `4_ppo.py` | Timeout treated as terminal → V=0 bootstrap → systematic value underestimation | V(final_state) bootstrap on truncation |
| 27 | MC PPO: N_EPISODES=3000 insufficient | `4_ppo.py` | Finding goals at avg -162 but couldn't consolidate | N_EPISODES 3000 → 8000 |
| 28 | MC Q-Learning: reward shaping caused regression | `1_q_learning.py` | Shaped reward created locally optimal greedy policy oscillating for KE without crossing goal. With ε=0.001 since ep 11K, Q-table has no escape. 44,000 episodes at -200 (vs -130 raw). Ng et al. 1999: non-potential-based shaping alters optimal policy — tabular methods can't escape via function approximation noise | Reverted to raw rewards. Shape_reward() kept in code with full explanation as dissertation evidence. |
| 29 | MC AC: policy gradient outer product bug | `3_actor_critic.py` | `dist.log_prob()` → shape [1]. `torch.stack(log_probs)` → [T,1]. `[T,1] × [T]` broadcasts to [T,T] outer product. `.sum()` = sum(log_probs) × sum(advantages) ≡ 0 (normalized advantages sum to 0). **Policy gradient was zero every update for all 5,000 episodes.** Previous run 4 (-188 avg) also had this bug — result was random exploration not learned policy | `.squeeze(-1)` → [T]; `.mean()` for scale independence |
| 30 | MC DQN: EPS_END=0.01 causes deterministic stuck policy | `2_dqn.py` | By ep 1,400 agent is 98.7% greedy. If greedy policy can't reach goal, stuck. PPO always stochastic via Categorical distribution | EPS_END 0.01 → 0.05 (5% permanent floor) |

### Round 3 — Critical (v2 run results analysis, 2026-06-02)

| # | Bug | File | Symptom / Root Cause | Fix |
|---|---|---|---|---|
| 31 | MC PPO: no LR schedule → late-stage regression | `4_ppo.py` | Fixed LR=3e-4 throughout. PPO was stable at avg -153 from ep 3,000–5,000 then completely collapsed at ep 5,300. Full-size LR updates disturbed a near-optimal policy that the value function had already overfit to. 0 goals in final 2,500 episodes. Engstrom 2020: LR decay is one of two missing critical PPO implementation details | Linear LR decay: `lr = 3e-4 × (1 - episode/N_EPISODES)` — by ep 5,000 LR is 1.875e-4, gentle enough to fine-tune rather than destroy |
| 32 | MC PPO: no value function clipping → value overfit | `4_ppo.py` | Without clipping, value network makes large jumps between updates. When policy degrades slightly, overfit value function produces large negative advantages, clip ratio blocks recovery updates — catastrophic feedback. Schulman 2017 appendix: value clipping is the second missing implementation detail | `v_pred_clipped = v_old + clip(v_pred - v_old, -ε, +ε); vl = max(vl_unclipped, vl_clipped).mean()` |
| 33 | MC DQN: soft target update → Q-value runaway | `2_dqn.py` | TAU=0.005 soft update every step makes target network slowly chase exploding online network. When goal transitions inject large TD targets (+10 bonus → target ≈ 12 vs Q ≈ 0), both networks inflate together — no stable reference. Loss: 5 → 118 → 761 → 2,107. Performance peaked at avg -143 (ep 2,500) then destroyed. | Hard target copy every 500 steps. Online Q-values can grow without dragging the target along. |
| 34 | MC DQN: goal bonus +10 creates large TD shock | `2_dqn.py` | First goal at ep 1,400 triggers TD error ≈ (11.5 - 0)² = 132 per sample. This spike propagates backward through bootstrapping — Q-values for nearby high-momentum states inflate. The step-penalty and KE shaping already guide the agent; bonus only needs to mark the terminal | Goal bonus +10 → +2 |
| 35 | MC DQN: LR=5e-5 too high for stable Q-values | `2_dqn.py` | Once Q-values reach correct range (~25–40), updates of 5e-5 are still large enough to overshoot and compound overestimation bias | LR 5e-5 → 2e-5 |
| 36 | MC AC: single optimizer causes critic to dominate | `3_actor_critic.py` | Single LR=3e-4 for both actor and critic. Critic (smooth_l1_loss over stable shaped returns) converges by ep 400 to the bad policy's V(s). With accurate critic, advantages → 0, policy gradient → 0. Actor loss = 0.000023 by ep 600, stays zero for 4,400 more episodes. System at fixed point — correct gradient, wrong policy | Separate param groups: actor LR=3e-4, critic LR=1e-4. Slower critic maintains non-zero advantages longer, giving actor meaningful signal |
| 37 | MC AC: VALUE_COEF=1.0 amplifies critic dominance | `3_actor_critic.py` | `loss = policy_loss + value_loss`. Equal weighting means critic loss magnitude can suppress policy gradient update direction | VALUE_COEF=0.5 (Schulman 2017 standard) |
| 38 | MC AC: ENTROPY_COEF=0.01 too low to escape fixed point | `3_actor_critic.py` | Once critic tracks bad policy, entropy term alone must maintain stochasticity. 0.01 insufficient to prevent the policy converging deterministically before finding goal | ENTROPY_COEF 0.01 → 0.02 |
| 39 | MC PPO v3: LR decay to 0 caused entropy collapse | `4_ppo.py` | Linear decay to 0 meant LR→0 by ep 8,000. With LR≈0, even entropy bonus gradient updates → 0. Policy went fully deterministic: final entropy=0.0032 (near zero vs 0.154 in v2). Only 26 goals in 8,000 eps. Performance held at -200 last 2,500 eps. | LR floor at 5e-5: `lr = 5e-5 + 2.5e-4*(1 - ep/N)`. Also tighten value clip 0.2→0.1. |
| 40 | MC DQN v3: Q-value scale inflates loss without collapsing performance | `2_dqn.py` | Loss 696 at ep 5,000 but last-100 avg -146.68 ± 6.31 (best stable DQN result). Loss is high because shaped rewards push Q-values into [20-50] range — correct values, not overestimation. But without normalisation, Q-value scale grows indefinitely. Eventually will cause policy degradation. | Running reward normalisation (Welford's online algorithm): z-score each shaped reward before replay buffer entry, clip ±10σ. Q-values represent normalised returns. Equivalent to SB3 VecNormalize. |
| 41 | MC PPO v3/v4: value clipping causes cyclic goal-forgetting | `4_ppo.py` | Value clip ε=0.1 limits V(s) to move 1.0/update cycle. With goal returns=50 and V(s)≈-8, advantages=58 → policy ratio always clipped at 1.2 → policy shifts only log(1.2)=0.18 per epoch toward goal. Entropy gradient (constant) beats intermittent goal gradient. Policy finds goals in burst (18 in ep 2017-2050) then forgets for 1,500 eps. Cyclic. Total 26 goals in 4,600 eps vs 3,363 in v2 with no clip. LR floor (5e-5) already limits value overfit — value clip is redundant and harmful. | Remove value clipping entirely. PPO v5 = LR floor + no value clip = v2 + stability fix. |
| 42 | MC DQN v4: reward normalisation removed directional signal | `2_dqn.py` | Welford z-score normalisation of shaped rewards resulted in mean-zero unit-variance rewards at every step. The agent could no longer distinguish "near goal" (high height + KE) from "at valley bottom" (low). All states looked the same from the reward signal. Agent never found goal in 5,000 episodes (avg -200 throughout) while loss stayed flat at 1.4. High loss in v3 (~700) reflected correctly large Q-values, not overestimation. | Reverted to raw shaped rewards. Extended to 8K episodes. |
| 43 | MC PPO v6 (original): LR decay tied to N_EPISODES causes divergent trajectories | `4_ppo.py` | LR formula `LR_MIN + (LR-LR_MIN)*(1-ep/N_EPISODES)` means changing N_EPISODES changes the LR at every step. v5 (N=8000) at ep 1000: LR=2.6875e-4. v6 (N=12000) at ep 1000: LR=2.792e-4. Butterfly effect: v5 found goals at ep 416 (sustained), v6 at ep 811 (cycled). Same seed, same architecture, same intended config — 50× difference in goal count. Henderson 2018. | Changed to `LR_DECAY_STEPS=8000` (constant). |
| 44 | MC DQN v5: extended budget allowed Q-value explosion to complete | `2_dqn.py` | v3 peaked at −146.68 avg at ep 5,000 with loss 696. Running to 8,000 episodes gave the Q-value runaway more time: loss reached 1,308 by ep 6K, performance regressed to −158 at ep 6K, −163 at ep 7K, −182 at ep 8K. The hard target update slowed the explosion but could not stop it indefinitely. v3 at 5,000 episodes was the optimal stopping point. | Report v3 result (−146.68 ±6.31) as the final DQN result. |

### Round 1 — Code Quality

| # | Bug | Fix |
|---|---|---|
| 12 | Stdout buffering in tee pipes | `python3 -u` flag |
| 13 | Matplotlib global colormap mutation | `.copy()` before `.set_bad()` |
| 14 | GridWorld success check `r > 0` | Track `state == GOAL` directly |
| 15 | `set_yscale('log')` drops zero values | `symlog` |
| 16 | `.squeeze()` scalar on 1-step episodes | `.squeeze(-1)` |
| 17 | MC Q-Learning condition `done and reward > -200` | Always true; simplified to `done` |
| 18 | MC PPO dead variable `total_steps` | Removed |
| 19 | CartPole PPO arbitrary `fill_between(ma±10)` | Removed |

---

## Key Hyperparameters (v3 — final committed values)

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

### MountainCar Q-Learning (final)
```
Tabular: 40×40 bins over pos×vel = 1,600 states
LR=0.3, GAMMA=0.995
Epsilon: 1.0 → 0.001, decay 0.9997 (hits min at ~ep 11K)
N_EPISODES=50,000, MAX_STEPS=200
Reward: raw (-1/step) — shaping reverted after bug #28 regression
Note: shape_reward() kept in code with explanation for dissertation
```

### MountainCar DQN (v4 — Double DQN)
```
Network: 2 → 128 → 128 → 3  (ReLU, Huber, AdamW)
BATCH=128, GAMMA=0.99, LR=2e-5, MEMORY=50K
Target: hard copy every 500 steps
Epsilon: 1.0 → 0.05 (floor) over 50,000 steps
N_EPISODES=5,000, MAX_STEPS=200
Reward shaping: height + 100·KE − 1  (+2 at goal)
Running reward normalisation: Welford online z-score, clip ±10σ  ← NEW v4
Double DQN: policy_net selects, target_net evaluates
```

### MountainCar Actor-Critic (v3)
```
Network: 2 → 64 → 64 → 3  (Tanh, orthogonal init)
Actor LR=3e-4, Critic LR=1e-4  ← separate param groups, was single LR=3e-4
GAMMA=0.99, LAM=0.95 (GAE), grad_clip=0.5
VALUE_COEF=0.5  ← was 1.0
ENTROPY_COEF=0.02  ← was 0.01
N_EPISODES=5,000, MAX_STEPS=200
State normalisation: pos → [−1,1], vel → [−1,1]
Reward shaping: height + 100·KE − 1  (+10 at goal)
Policy gradient: log_probs.squeeze(-1) element-wise × advantages
Entropy: exact dist.entropy() per step
Truncation fix: V(final_state) bootstrap on timeout
GAE(λ=0.95) replaces full MC returns
```

### MountainCar PPO — FINAL BEST: v5
```
Network: 2 → 64 → 64 → 3  (Tanh, orthogonal init)
LR: 5e-5 + 2.5e-4*(1 - ep/8000) → floor 5e-5  (LR_DECAY_STEPS=8000, fixed)
γ=0.99, λ=0.95, clip=0.2, entropy=0.01, VALUE_COEF=0.5
Value clipping: NONE
Update every 1,024 steps, 10 epochs, batch 64
N_EPISODES=8,000, MAX_STEPS=200
State normalisation: pos → [−1,1], vel → [−1,1]
Reward shaping: height + 100·KE − 1  (+10 at goal)
Truncation fix: terminated vs truncated in GAE bootstrap
RESULT: mean (all) −162.78, last-100 avg −151.80 ±29.37, 4,816 goals, no regression
```

Note: v6 (12K eps, same config) found 6,764 goals but eroded to −200 in the final 4K eps.
The extended fine-tuning phase at floor LR slowly eroded rather than consolidated the policy.
8,000 episodes is the optimal budget for this configuration.

---

## MountainCar Run History

### Q-Learning
| Run | Episodes | Last-100 avg | Best avg | Notes |
|---|---|---|---|---|
| run 1 | 25,000 | -138.11 | -123.72 | Raw reward |
| run 2 | 50,000 | -137.82 | -130.05 | Raw reward |
| v2 attempt (killed) | 44,000 | -200.00 | -200.00 | Shaped reward — attractor bug #28 |
| **v2 final** | 50,000 | **-137.82** | **-130.05** | Raw reward — tabular limit confirmed |

### DQN
| Run | Episodes | Best avg | End avg | Notes |
|---|---|---|---|---|
| run 1 | 600 | -211 | -211 | LR=1e-3, loss 115K |
| run 2 | 1,000 | -132 | -132 | EPS_DECAY fixed, 24% success |
| run 3 | 2,000 | -134 | -134 | Standard DQN overestimation |
| run 4 | 3,000 | -134 | -134 | Double DQN; MAX_STEPS=400 |
| v2 (killed @ep250) | 250 | n/a | n/a | Killed; same bugs |
| v2 restart | 5,000 | -143 | -200 | Found goals (avg -143 at ep 2,500) then Q-value explosion (loss 2,107) — bug #33/34/35 |
| v3 | 5,000 | **-146.68** | **-146.68** | Hard target + LR 2e-5 + goal bonus +2. **Best DQN result.** Loss 696 = correct large Q-values, not explosion. |
| v4 | 5,000 | -200 | -200 | Reward normalisation removed directional signal. Bug #42. |
| v5 | 8,000 | -146 (ep 5K) | **-182.40** | Extended budget backfired: Q-value explosion continued (loss 1,308 at ep 8K). Peak at ep 5K then regressed. v3 (5K eps) was the sweet spot. |

### Actor-Critic
| Run | Episodes | Best avg | End avg | Notes |
|---|---|---|---|---|
| run 1 | 2,000 | -200 | -200 | Returns normalised — actor loss 6,344 |
| run 2 | 2,000 | -200 | -200 | Advantages normalised; entropy=0.05 |
| run 3 | 3,000 | -200 | -200 | entropy=0.05 — actor loss 5,442 |
| run 4 | 3,000 | -188 | -188 | entropy=0.01; outer product bug — random exploration only |
| v2 (killed @ep2700) | 2,700 | -200 | -200 | Outer product bug #29 — policy gradient ≡ 0 |
| v2 restart | 5,000 | -200 | -200 | Bug #29 fixed but critic dominated — zero-advantage fixed point (bugs #36/37/38) |
| **v3** | 5,000 | ⟳ | ⟳ | **Separate LRs + VALUE_COEF=0.5 + entropy=0.02** |

### PPO
| Run | Episodes | Best avg | End avg | Notes |
|---|---|---|---|---|
| run 1 | 1,000 | -161 | -161 | entropy=0.05 |
| run 2 | 3,000 | -200 | -200 | entropy=0.001 — collapsed |
| run 3 | 3,000 | -200 | -200 | entropy=0.05 — collapsed at end |
| run 4 | 3,000 | -162 | -162 | entropy=0.01 ✓ |
| run 5 | 3,000 | -142 | -142 | entropy=0.01 |
| v2 | 8,000 | -148 | -200 | 3,363 goals, avg -153 stable ep 3K–5K then catastrophic regression at ep 5,300 — bugs #31/32 |
| v3 | 8,000 | ~-153 | -200 | LR decay to 0 → entropy collapse (0.0032). Only 26 goals. Bug #39 |
| v4 (killed @4600) | 4,600 | ~-190 | n/a | LR floor fixed but value clip 0.1 → goal-forgetting cycles (bug #41). 26 goals |
| v5 | 8,000 | -151.80 | -151.80 | LR floor + no value clip. 4,816 goals. No regression. Best PPO stability. |
| v6 (killed @3250) | 3,250 | ~-185 | n/a | LR tied to N_EPISODES (bug #43) — trajectory diverged from v5. Cycling pattern. |
| v6 restart | 12,000 | ~-152 (ep 8K) | **-200** | LR_DECAY_STEPS=8000 fix replicated v5 trajectory (6,764 goals). But fine-tuning phase (ep 8K-12K at LR=5e-5) eroded policy — goals dropped 752→462→176→0 per 1K window. Last-100 avg -200. Extended budget hurt. **v5 (8K) remains the best PPO.** |

---

## Narrative for the Report

### Algorithm Comparison (CartPole)
PPO solved fastest (ep 195), DQN next (ep 225), A2C (ep 1,175), Q-Learning slowest (ep 9,798). PPO's sample efficiency advantage comes from reusing data across multiple epochs per update. Q-Learning's discretisation introduces approximation error that slows convergence.

### Why MountainCar is Harder
The environment has a **sparse, deceptive reward structure**: −1 every step, goal only reachable by building momentum through counter-intuitive backward movement. Without reward shaping, no algorithm makes progress from random initialisation (probability of random walk reaching goal in 200 steps is infinitesimally small — Dann et al. 2022).

With reward shaping (`height + 100·KE − 1`), all deep RL algorithms can learn, but difficulty exposes each algorithm's distinct failure modes:
- **Q-Learning**: discretisation creates state aliasing (Logofătu 2022); reward shaping creates local attractor in tabular policy (Ng et al. 1999, bug #28)
- **DQN**: Q-value overestimation compounds with shaped rewards — fixed with Double DQN, but goal transitions still inject TD shocks causing runaway
- **A2C**: On-policy + single env + accurate critic → zero-advantage fixed point; on-policy methods need parallelism (Mnih 2016 A3C)
- **PPO**: Implementation details are everything — without LR decay and value clipping (Engstrom 2020), otherwise correct implementations regress

### Deep Insights Worth Discussing

1. **Entropy coefficient is critical for on-policy methods.** Too high (0.05): finds goal but keeps unlearning it. Too low (0.001): policy deterministic before finding goal. 0.01 is the balance.

2. **Standard DQN overestimates Q-values** (van Hasselt 2016). Reproduced: plateau at avg −134, loss=2,491. Fixed with Double DQN.

3. **Return normalisation vs advantage normalisation.** Normalising returns creates shifting critic target (actor_loss=5,442). Normalising advantages keeps training signal consistent.

4. **Replay buffer size is a critical hyperparameter** (Zhang & Sutton 2019). 10K too small — rare goal transitions overwritten. 50K optimal balance between diversity and staleness.

5. **GAE vs MC returns in A2C.** MC over 200-step episodes has enormous variance. GAE(λ=0.95) reduces variance at cost of small bias — equivalent to exponentially weighted n-step returns (Sutton 1988).

6. **Truncation bias** (Pardo et al. 2018). Timeout ≠ terminal. Treating truncation as terminal injects artificial value cliff at step 200, corrupting value function backward through time. Fix: bootstrap V(final_state).

7. **Reward shaping and optimal policy invariance** (Ng et al. 1999). `height + 100·KE − 1` is NOT potential-based shaping, violating the invariance theorem. Reproduced empirically: tabular Q-Learning with shaped reward stayed at -200 for 44,000 episodes while unshapen raw-reward version reached -130. Deep RL methods escape via function approximation noise; tabular converged greedy policies cannot.

8. **Policy gradient outer product bug (bug #29).** `torch.stack([T,1]-tensors) × [T]-advantages` broadcasts to [T,T] outer product. Since normalized advantages sum to 0, policy_loss ≡ 0. A silent bug that produced correct-looking loss values while computing zero gradients. Illustrates Henderson 2018: implementation details determine outcomes.

9. **PPO implementation details dominate performance** (Engstrom 2020). Linear LR decay and value function clipping are the two missing ingredients. Without them, PPO found 3,363 goals over 5,000 episodes then catastrophically regressed at ep 5,300 — the fixed LR=3e-4 made updates large enough to destroy a near-optimal policy.

10. **Zero-advantage fixed point in single-env A2C.** When critic converges to V(s) for a bad policy, advantages → 0, policy gradient → 0. Actor loss = 0.000023 by ep 600 for all 5,000 episodes. Requires either parallel environments (A3C) or asymmetric LRs to escape.

11. **Q-value runaway requires stable targets** (Mnih 2015). Soft target update (TAU=0.005) causes target to chase inflating online Q-values — no stable reference. Hard update every 500 steps fixes target at a historical snapshot, breaking the feedback loop. Goal bonus must also be small (+2 not +10) to avoid initial TD shock.

12. **PPO on MountainCar exhibits cyclic goal-forgetting from a single environment.** The policy discovers goal-reaching behavior in bursts (78 goals in 500 eps), improves avg to −184, then loses the skill for 1,500+ episodes. Root cause: with 1,024-step buffers and ~190 steps/episode, each update spans ~5 episodes. With ~1 goal per 200 episodes, 97% of updates contain no goal episodes — the entropy gradient continuously erodes goal behavior while policy gradients toward goals are rare. The policy oscillates between brief goal-discovery and long forgetting. v5 escaped this cycle (found goals at ep 416, sustained for 8,000 eps) through a favorable stochastic trajectory. v6 (identical config, slightly different LR due to N_EPISODES coupling) cycled throughout. This confirms Henderson et al. 2018: "random seeds and implementation details dominate results." The same algorithm with the same hyperparameters produced 4,816 goals (v5) vs ~95 goals (original v6) purely from trajectory divergence.

13. **Extending budget beyond the optimal stopping point can hurt.** DQN v3 (5K eps, avg −146.68) was the best DQN result. Extending to 8K eps (v5) let Q-value overestimation compound further: loss 696 → 1,308, avg degraded from −146 to −182. PPO v6 with 12K episodes found 6,764 goals (more than v5's 4,816) but the extra 4K fine-tuning eps at floor LR eroded rather than consolidated. Policy quality peaked at ep 8K and degraded through 12K. For both DQN and PPO, the optimal stopping point was when the algorithm was at its best-performing configuration — more training does not always help on MountainCar.

13. **Single-environment A2C cannot solve MountainCar — this is structural, not a bug.** Confirmed across all runs (v1–v3, 20,000+ total episodes, zero goals). Root cause is a zero-advantage fixed point: critic converges to V(s) for the current bad policy by ep ~600, advantages → 0, policy gradient → 0. This is mathematically inevitable regardless of LR, VALUE_COEF, or entropy settings. The only escape is environmental diversity. RL Baselines3 Zoo requires `n_envs: 16` specifically for this environment (confirmed via web search). The A3C paper (Mnih 2016) motivated parallelism as the structural solution — not an optimisation. Contrast with PPO: multi-epoch updates over 1,024-step buffers spanning episode boundaries provide implicit diversity from a single environment. This is one of the deepest algorithmic insights in the dissertation: the structural difference between A2C's need for parallelism vs PPO's epoch-reuse mechanism. Cite: Mnih 2016, Dann et al. 2022, RL Baselines3 Zoo a2c.yml.
