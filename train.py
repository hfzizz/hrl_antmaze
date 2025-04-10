import os
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
from datetime import datetime
import time
from agents.ppo_agent import PPOAgent

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB / {torch.cuda.memory_reserved(0)/1024**2:.2f}MB")

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Register the robotics environments
try:
    gym.register_envs(gymnasium_robotics)
except:
    print("Environments already registered")

def train():
    # Create logs directory
    log_dir = f"logs/ppo_antmaze_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

     # Environment setup
    env_name = "AntMaze_UMaze-v5"
    print(f"Creating environment {env_name}...")
    
    # Make sure the environment exists in the registry
    all_envs = gym.envs.registry.keys()
    if env_name not in all_envs:
        print(f"Available environments: {[env for env in all_envs if 'Ant' in env]}")
        raise ValueError(f"Environment {env_name} not found in registry")
    
    env = gym.make(env_name, render_mode="human")
    print(f"Environment created successfully")
    
    # Print space information
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Extract the observation dimension from the environment
    if isinstance(env.observation_space, gym.spaces.Dict):
        # Use the 'observation' key for the state dimensions
        state_dim = env.observation_space['observation'].shape[0]
        print(f"Using 'observation' key from Dict space: {state_dim}")
    else:
        state_dim = env.observation_space.shape[0]
        print(f"Using direct observation space: {state_dim}")

    action_dim = env.action_space.shape[0]
    print(f"State dim: {state_dim}, Action dim: {action_dim}") 

    # Set training hyperparameters
    max_episodes = 10000
    max_timesteps = 1000
    update_timestep = 4000  # Update policy every n timesteps
    log_interval = 20       # Print avg reward after n episodes
    save_interval = 100     # Save model every n episodes
    
    # Initialize PPO agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_std_init=0.6,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        K_epochs=80,
        update_timestep=update_timestep,
        gae_lambda=0.95,
        device=device
    )
    
    # Logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    
    # Training loop
    print(f"Starting training on {env_name}...")
    
    for i_episode in range(1, max_episodes+1):
        obs_dict, _ = env.reset()
        # Extract observation from the environment
        state = obs_dict['observation']
        episode_reward = 0
        
        for t in range(max_timesteps):
            # Select action from policy
            action = agent.select_action(state)
            
            # Take action in environment
            next_obs_dict, reward, terminated, truncated, _ = env.step(action)
            next_state = next_obs_dict['observation']
            done = terminated or truncated
            
            # Update reward and done flag in agent's memory
            agent.update_reward_done(reward, done)
            
            state = next_state
            episode_reward += reward
            timestep += 1
            
            # Break if episode is done
            if done:
                break
        
        # Update running reward
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        
        # Log stats
        if i_episode % log_interval == 0:
            print(f"Episode {i_episode}\tLast reward: {episode_reward:.2f}\tAverage reward: {running_reward:.2f}")
            
            # Log to file
            with open(os.path.join(log_dir, "training_log.csv"), "a") as f:
                if i_episode == log_interval:  # Write header on first log
                    f.write("episode,reward,avg_reward\n")
                f.write(f"{i_episode},{episode_reward:.2f},{running_reward:.2f}\n")
        
        # Save model
        if i_episode % save_interval == 0:
            agent.save(os.path.join(model_dir, f"ppo_model_ep{i_episode}.pt"))
            
        # Decay action std for better exploitation
        if i_episode % 500 == 0:
            agent.decay_action_std(min_action_std=0.1, decay_rate=0.05)
    
    # Save final model
    agent.save(os.path.join(model_dir, "ppo_model_final.pt"))
    env.close()
    
    print("Training completed!")

if __name__ == "__main__":
    train()