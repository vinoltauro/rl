#!/bin/bash
cd /home/taurovinol/rl

echo "=== Starting MC PPO (entropy=0.01) ==="
python3 -u mountaincar/4_ppo.py 2>&1 | tee mc_ppo5_run.log
echo "=== MC PPO done. Starting MC DQN (Double DQN, LR=5e-5) ==="

python3 -u mountaincar/2_dqn.py 2>&1 | tee mc_dqn5_run.log
echo "=== All done ==="
