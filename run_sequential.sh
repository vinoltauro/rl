#!/bin/bash
cd /home/taurovinol/rl

echo "=== Starting MC DQN ==="
python3 -u mountaincar/2_dqn.py 2>&1 | tee mc_dqn3_run.log
echo "=== MC DQN done. Starting MC AC ==="

python3 -u mountaincar/3_actor_critic.py 2>&1 | tee mc_ac3_run.log
echo "=== MC AC done. Starting MC PPO ==="

python3 -u mountaincar/4_ppo.py 2>&1 | tee mc_ppo3_run.log
echo "=== All MountainCar runs done ==="

echo "=== Starting MC DQN (2000 eps) ==="
python3 -u mountaincar/2_dqn.py 2>&1 | tee mc_dqn4_run.log
echo "=== All done ==="
