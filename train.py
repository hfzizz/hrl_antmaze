import os
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

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
    log_dir = f"logs/sb3_ppo_antmaze_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    
    # Create vectorized training environment with normalization
    n_envs = 4  # Number of parallel environments
    env = make_vec_env(env_name, n_envs=n_envs, seed=0, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create evaluation environment with matching structure (vectorized and normalized)
    # Use DummyVecEnv for a single environment with vectorized interface
    eval_env = make_vec_env(env_name, n_envs=1, seed=0)
    # Add normalization with same parameters
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    # If you want rendering, unfortunately you'll need to use a different approach
    # as rendering with vectorized environments is more complex
    
    
    print(f"Environment created successfully")
    
    # Print space information
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Set training hyperparameters
    total_timesteps = 1000000  # Total timesteps for training
    log_interval = 10          # Print logs every n updates
    save_interval = 50000      # Save model every n steps
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_interval,
        save_path=model_dir,
        name_prefix="ppo_antmaze"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best_model"),
        log_path=log_dir,
        eval_freq=20000,
        deterministic=True,
        render=False
    )
    
    # Initialize the PPO agent with improved parameters
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=0.0001,
        n_steps=2048,
        batch_size=256,  # Increased batch size for better learning
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Added some entropy for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        device=device,
        verbose=1
    )
    
    print(f"Starting training on {env_name} with {n_envs} parallel environments...")
    
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=log_interval
    )
    
    # Save the final model and normalization stats
    final_model_path = os.path.join(model_dir, "ppo_antmaze_final")
    model.save(final_model_path)
    stats_path = os.path.join(model_dir, "vec_normalize_stats")
    env.save(stats_path)
    
    # Final evaluation
    mean_reward, std_reward = evaluate_policy(
        model.policy,
        eval_env,
        n_eval_episodes=5,
        deterministic=True
    )
    print(f"Final evaluation: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    
    print("Training completed!")

    return model, env_name

# After training, create a separate environment for visualization
def visualize_agent(model, env_name, num_episodes=3):
    vis_env = gym.make(env_name, render_mode="human")
    for _ in range(num_episodes):
        obs, _ = vis_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = vis_env.step(action)
            done = terminated or truncated
    vis_env.close()

if __name__ == "__main__":
    trained_model, env_name = train()
    visualize_agent(trained_model, env_name)