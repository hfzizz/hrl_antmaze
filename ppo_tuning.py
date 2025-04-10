import os
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
from datetime import datetime
import optuna
from agents.ppo_agent import PPOAgent

# Setup
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Register environments
try:
    gym.register_envs(gymnasium_robotics)
except:
    print("Environments already registered")

# Create base log directory
base_log_dir = f"logs/optuna_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(base_log_dir, exist_ok=True)

# Environment setup function
def create_env():
    env_name = "AntMaze_UMaze-v5"
    env = gym.make(env_name)
    return env

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters using Optuna
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    eps_clip = trial.suggest_float("eps_clip", 0.1, 0.3)
    action_std = trial.suggest_float("action_std", 0.3, 1.0)
    k_epochs = trial.suggest_int("k_epochs", 10, 80)
    
    # Create a specific log directory for this trial
    trial_log_dir = os.path.join(base_log_dir, f"trial_{trial.number}")
    os.makedirs(trial_log_dir, exist_ok=True)
    
    # Log the suggested parameters
    with open(os.path.join(trial_log_dir, "params.txt"), "w") as f:
        f.write(f"lr: {lr}\n")
        f.write(f"gamma: {gamma}\n")
        f.write(f"eps_clip: {eps_clip}\n")
        f.write(f"action_std: {action_std}\n")
        f.write(f"k_epochs: {k_epochs}\n")
    
    # Create environment
    env = create_env()
    
    # Get state and action dimensions
    if isinstance(env.observation_space, gym.spaces.Dict):
        state_dim = env.observation_space['observation'].shape[0]
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create agent with suggested hyperparameters
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_std_init=action_std,
        lr=lr,
        gamma=gamma,
        eps_clip=eps_clip,
        K_epochs=k_epochs,
        update_timestep=4000,
        gae_lambda=0.95,
        device=device
    )
    
    # Run a training session
    n_episodes = 200  # Reduced for faster tuning
    running_reward = 0
    rewards_log = []
    
    # Training loop
    for i_episode in range(1, n_episodes + 1):
        obs_dict, _ = env.reset()
        state = obs_dict['observation']
        episode_reward = 0
        
        for t in range(1000):  # Max steps per episode
            action = agent.select_action(state)
            
            next_obs_dict, reward, terminated, truncated, _ = env.step(action)
            next_state = next_obs_dict['observation']
            done = terminated or truncated
            
            # Add a shaping reward to help with the sparse reward problem
            # Use distance to goal as a shaping reward
            goal_pos = next_obs_dict['desired_goal']
            agent_pos = next_state[:2]  # Assuming first two values are x,y position
            distance_to_goal = np.linalg.norm(agent_pos - goal_pos)
            shaped_reward = reward - 0.01 * distance_to_goal  # Small penalty for distance
            
            agent.update_reward_done(shaped_reward, done)
            
            state = next_state
            episode_reward += reward  # Track original reward for evaluation
            
            if done:
                break
        
        # Report intermediate results to Optuna
        if i_episode % 50 == 0:
            trial.report(running_reward, i_episode)
            
            # Check if the trial should be pruned
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        # Update running reward
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        rewards_log.append((i_episode, episode_reward, running_reward))
        
        # Log periodically
        if i_episode % 20 == 0:
            print(f"Trial {trial.number}, Episode {i_episode}/{n_episodes}: Reward={episode_reward:.2f}, Avg={running_reward:.2f}")
        
        # Decay exploration
        if i_episode % 50 == 0:
            agent.decay_action_std(min_action_std=0.1, decay_rate=0.05)
    
    # Save results
    with open(os.path.join(trial_log_dir, "rewards.csv"), "w") as f:
        f.write("episode,reward,avg_reward\n")
        for ep, rew, avg in rewards_log:
            f.write(f"{ep},{rew:.2f},{avg:.2f}\n")
    
    # Save the model
    agent.save(os.path.join(trial_log_dir, "final_model.pt"))
    
    # Return the final running reward as the objective value
    return running_reward

def main():
    print("Starting Optuna hyperparameter optimization...")
    
    # Create a study object with pruning enabled
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100)
    study = optuna.create_study(
        direction="maximize", 
        pruner=pruner,
        study_name="ppo_antmaze_optimization"
    )
    
    # Run the optimization
    study.optimize(objective, n_trials=25, timeout=36000)  # 10 hours max
    
    print("Optimization finished.")
    
    # Print summary statistics
    print("\n===== OPTUNA RESULTS =====")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.3f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save the study results
    with open(os.path.join(base_log_dir, "best_params.txt"), "w") as f:
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best value: {study.best_value:.3f}\n\n")
        f.write("Best hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")
        
        # Add this at the end of your main function
    import optuna.visualization as vis

    # Plot optimization history
    fig = vis.plot_optimization_history(study)
    fig.write_html(os.path.join(base_log_dir, "optimization_history.html"))

    # Plot parameter importances
    fig = vis.plot_param_importances(study)
    fig.write_html(os.path.join(base_log_dir, "param_importances.html"))

    # Plot parameter relationships
    fig = vis.plot_parallel_coordinate(study)
    fig.write_html(os.path.join(base_log_dir, "parallel_coordinate.html"))

if __name__ == "__main__":
    main()