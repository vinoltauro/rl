import gymnasium as gym
import torch
import time
import sys

class ActorCritic(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCritic, self).__init__()
        
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, action_size),
            torch.nn.Softmax(dim=-1)
        )
        
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

def watch_agent(model_path=None, n_episodes=5, delay=0.02):
    env = gym.make('CartPole-v1', render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ActorCritic(state_size, action_size).to(device)
    
    if model_path:
        try:
            policy.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✓ Loaded model from: {model_path}")
        except:
            print(f"✗ Could not load model from: {model_path}")
            print("  Running with randomly initialized policy")
    else:
        print("No model path provided - running with random policy")
    
    policy.eval()
    
    print("\n" + "="*70)
    print("WATCHING AGENT PLAY CARTPOLE")
    print("="*70)
    print("Press Ctrl+C to stop\n")
    
    episode_rewards = []
    
    try:
        for episode in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            
            print(f"Episode {episode + 1}/{n_episodes}...", end=" ", flush=True)
            
            for step in range(500):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    action_probs, _ = policy(state_tensor)
                
                action = action_probs.argmax().item()
                
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                time.sleep(delay)
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(total_reward)
            print(f"Reward: {total_reward:.0f}, Steps: {steps}")
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    env.close()
    
    if episode_rewards:
        print("\n" + "="*70)
        print(f"Average Reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
        print(f"Episodes: {episode_rewards}")
        print("="*70)

if __name__ == "__main__":
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    watch_agent(model_path=model_path, n_episodes=5, delay=0.02)