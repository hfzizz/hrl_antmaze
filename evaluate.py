import os
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Register the robotics environments
try:
    gym.register_envs(gymnasium_robotics)
except:
    print("Environments already registered")

def load_model(model_path, env_name, stats_path=None):
    """Load a trained model and create the environment."""
    print(f"Loading model from: {model_path}")
    
    # Create environment for evaluation
    env = gym.make(env_name)
    
    # If vectorized environment stats are provided, create a normalized environment
    if stats_path and os.path.exists(stats_path):
        print(f"Loading normalization stats from: {stats_path}")
        # Create a vectorized environment
        vec_env = make_vec_env(env_name, n_envs=1)
        # Load the saved stats
        vec_env = VecNormalize.load(stats_path, vec_env)
        # Don't update the normalization statistics during evaluation
        vec_env.training = False
        vec_env.norm_reward = False
        eval_env = vec_env
    else:
        print("No normalization stats found, using raw environment")
        eval_env = env
    
    # Load the trained model
    model = PPO.load(model_path, env=eval_env)
    
    return model, eval_env

def evaluate_model(model, env, num_episodes=10):
    """Evaluate the model's performance over multiple episodes."""
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=num_episodes,
        deterministic=True,
        render=False
    )
    print(f"Evaluation over {num_episodes} episodes: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward

def visualize_model(model, env_name, num_episodes=3):
    """Visualize the model in a human-renderable environment."""
    print(f"Visualizing model on {env_name} for {num_episodes} episodes")
    # Create a separate environment for visualization with rendering enabled
    vis_env = gym.make(env_name, render_mode="human")
    
    for episode in range(num_episodes):
        print(f"Episode {episode+1}/{num_episodes}")
        obs, _ = vis_env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            # Get the model's action
            action, _ = model.predict(obs, deterministic=True)
            # Execute the action
            obs, reward, terminated, truncated, info = vis_env.step(action)
            episode_reward += reward
            step_count += 1
            done = terminated or truncated
            
        print(f"Episode complete: Reward = {episode_reward:.2f}, Steps = {step_count}")
    
    vis_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained PPO model on AntMaze')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='Path to the trained model file (.zip)')
    parser.add_argument('--stats_path', type=str, default=None,
                        help='Path to the normalization statistics file')
    parser.add_argument('--env_name', type=str, default="AntMaze_UMaze-v5",
                        help='Environment name')
    parser.add_argument('--num_eval_episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
    parser.add_argument('--num_vis_episodes', type=int, default=3,
                        help='Number of episodes for visualization')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the agent behavior')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate the agent performance')
    
    args = parser.parse_args()
    
    # Find the latest model if none specified
    if args.model_path is None:
        # Look in the logs directory for the best model
        log_dirs = [d for d in os.listdir("logs") if d.startswith("sb3_ppo_antmaze")]
        if log_dirs:
            # Sort by date (newest first)
            log_dirs.sort(reverse=True)
            best_model_path = os.path.join("logs", log_dirs[0], "models", "best_model", "best_model.zip")
            if os.path.exists(best_model_path):
                args.model_path = best_model_path
                print(f"Using latest best model: {args.model_path}")
                
                # Try to find the corresponding stats file
                stats_path = os.path.join("logs", log_dirs[0], "models", "vec_normalize_stats")
                if os.path.exists(stats_path):
                    args.stats_path = stats_path
        
        if args.model_path is None:
            raise ValueError("No model specified and couldn't find a best model automatically")
    
    model, env = load_model(args.model_path, args.env_name, args.stats_path)
    
    # Evaluate and/or visualize based on arguments
    if args.evaluate or not args.visualize:  # Default to evaluation if nothing specified
        evaluate_model(model, env, args.num_eval_episodes)
    
    if args.visualize:
        visualize_model(model, args.env_name, args.num_vis_episodes)