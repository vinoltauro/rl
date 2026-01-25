# MountainCar Q-Learning - Quick Start Guide

## What You'll Do

Train an AI agent to drive a car up a steep hill by building momentum!

---

## Option 1: Automated Setup (Easiest - Mac/Linux)

```bash
# 1. Download all files to a folder called 'mountaincar_qlearning'
# 2. Open terminal in that folder
# 3. Run this one command:

chmod +x setup_and_run.sh && ./setup_and_run.sh
```

That's it! Script does everything automatically.

---

## Option 2: Manual Setup (Works on All Systems)

### Step 1: Create Project Directory
```bash
mkdir mountaincar_qlearning
cd mountaincar_qlearning
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
```

### Step 3: Activate Virtual Environment

**Mac/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```cmd
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### Step 4: Install Required Packages
```bash
pip install gymnasium numpy matplotlib
```

### Step 5: Add the Code File

Save `mountaincar_qlearning.py` in the `mountaincar_qlearning` folder.

### Step 6: Run the Training
```bash
python mountaincar_qlearning.py
```

---

## What to Expect

### Training Output:
```
==========================================================
TRAINING MOUNTAINCAR Q-LEARNING AGENT
==========================================================
Episode   100/10000 | Avg Reward: -200.00 | Avg Steps: 200.00 | Epsilon: 0.951
Episode   200/10000 | Avg Reward: -200.00 | Avg Steps: 200.00 | Epsilon: 0.905
Episode   300/10000 | Avg Reward: -200.00 | Avg Steps: 200.00 | Epsilon: 0.861
...
[Around episode 3000-5000, you'll see improvement:]
...
Episode  5000/10000 | Avg Reward: -165.32 | Avg Steps: 165.32 | Epsilon: 0.082
Episode  6000/10000 | Avg Reward: -142.45 | Avg Steps: 142.45 | Epsilon: 0.050
Episode  7000/10000 | Avg Reward: -118.67 | Avg Steps: 118.67 | Epsilon: 0.030
Episode  8000/10000 | Avg Reward: -105.23 | Avg Steps: 105.23 | Epsilon: 0.018
Episode  9000/10000 | Avg Reward:  -98.45 | Avg Steps:  98.45 | Epsilon: 0.011
Episode 10000/10000 | Avg Reward:  -96.78 | Avg Steps:  96.78 | Epsilon: 0.010
==========================================================
TRAINING COMPLETED
==========================================================
```

### Generated Files:
- `mountaincar_training_results.png` - 6 plots showing training progress

### Test Results:
```
Testing trained agent...
Test Episode  1: Reward =  -98.00, Steps =  98
Test Episode  2: Reward =  -95.00, Steps =  95
Test Episode  3: Reward = -102.00, Steps = 102
...
Average Test Reward: -98.50
Average Test Steps: 98.50
Agent successfully learned to reach the goal efficiently!
```

---

## Understanding MountainCar

### The Problem:
```
        Goal! 🚩
          /  \
         /    \____
        /          \___
       /                \___     
      /                     \___🚗___
     /_____________________________________\
```

- Car starts in a valley
- Goal: Reach the flag on the right hill
- Problem: Engine too weak to drive straight up!
- Solution: Rock back and forth to build momentum

### Key Differences from CartPole:

| Feature | CartPole | MountainCar |
|---------|----------|-------------|
| Goal | Balance pole | Reach flag |
| Reward | +1 per step (good!) | -1 per step (sparse!) |
| Actions | 2 (left, right) | 3 (left, none, right) |
| Success | 195+ steps average | <110 steps average |
| Difficulty | Medium | Hard |
| Training time | ~3k episodes | ~10k episodes |

### Why It's Harder:
1. **Sparse rewards**: Every action gives -1, no positive feedback
2. **Counter-intuitive**: Must go LEFT (away from goal) to build momentum
3. **Strategic**: Requires planning multiple steps ahead
4. **Delayed reward**: Good actions now pay off 10+ steps later

---

## Interpreting Results

### Reward Values:
- `-200`: Bad (timeout, didn't reach goal)
- `-150`: Getting better (reached goal slowly)
- `-110`: Good! (reached goal efficiently)
- `-95`: Excellent! (near optimal)

**Remember:** Less negative = better!

### Success Criteria:
✓ Average reward (last 100 episodes): -95 to -110  
✓ Test performance: < 110 steps consistently  
✓ Plots show clear improvement trend  

---

## Troubleshooting

### "Always getting -200 rewards"
- **Normal for first 3000-5000 episodes!**
- Agent needs time to stumble upon success
- Keep training, breakthrough will come

### "Training is slow"
- Expected: ~5-10 minutes for 10,000 episodes
- If >20 minutes, check if computer is doing other tasks

### "ImportError: No module named gymnasium"
- Make sure venv is activated (see `(venv)` in prompt)
- Run: `pip install gymnasium numpy matplotlib`

### "Results are inconsistent"
- Normal! MountainCar is harder than CartPole
- Some randomness in exploration
- Average over 100 episodes is what matters

---

## What the Plots Show

### Plot 1: Reward per Episode
- Raw rewards for each episode
- Should trend upward (less negative)
- Will be noisy (lots of variance)

### Plot 2: Moving Average Reward
- Smoothed trend
- Should clearly show improvement
- Target: reach -110 or better

### Plot 3: Episode Length (Steps)
- How many steps to reach goal
- Should trend downward
- Target: under 110 steps

### Plot 4: Moving Average Steps
- Smoothed steps trend
- Clearer view of learning

### Plot 5: Epsilon Decay
- Exploration rate over time
- Starts at 1.0, decays to 0.01

### Plot 6: Reward Distribution
- Histogram of all episode rewards
- Should shift right (less negative) over training

---

## Files You Have

1. **mountaincar_qlearning.py** (5.2 KB)
   - Clean code, no comments
   - Ready to run

2. **MOUNTAINCAR_QLEARNING_GUIDE.txt** (22 KB)
   - Complete conceptual explanation
   - Comparison with CartPole
   - Troubleshooting guide

3. **setup_and_run.sh** (Optional)
   - Automated setup script
   - Mac/Linux only

---

## Quick Commands Reference

```bash
# Setup (one time)
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install gymnasium numpy matplotlib

# Run training
python mountaincar_qlearning.py

# Deactivate venv when done
deactivate
```

---

## Time Estimates

- Setup: 2 minutes
- Training: 5-10 minutes
- Reviewing results: 5 minutes
- **Total: ~15-20 minutes**

---

## Success Checklist

After running, you should have:

- [x] Training completed (10,000 episodes)
- [x] Plot file: `mountaincar_training_results.png`
- [x] Final average reward: -95 to -110
- [x] Test performance: < 110 steps average
- [x] Clear upward trend in plots

If all checked, you've successfully trained a MountainCar agent! 🎉

---

## Next Steps (Optional)

1. **Experiment**: Try different hyperparameters
2. **Compare**: How does it differ from CartPole?
3. **Analyze**: Why did certain strategies work?
4. **Visualize**: Watch the agent in action (add visualization code)

---

## Need Help?

Read the comprehensive guide: `MOUNTAINCAR_QLEARNING_GUIDE.txt`

It covers:
- Detailed problem explanation
- Why MountainCar is hard
- Expected training phases
- Troubleshooting common issues
- Comparison with CartPole

---

Good luck! Remember: seeing -200 for thousands of episodes is NORMAL. Be patient! 🚗⛰️
