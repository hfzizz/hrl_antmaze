import os
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics
from datetime import datetime
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

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
def create_env(n_envs=4):
    env_name = "AntMaze_UMaze-v5"
    # Create vectorized training environment with normalization
    env = make_vec_env(env_name, n_envs=n_envs, seed=0, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    return env, env_name

# Create a proper Optuna callback that extends BaseCallback
class OptunaCallback(BaseCallback):
    def __init__(self, trial, eval_env, eval_freq=10000, verbose=0):
        super().__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
    
    def _init_callback(self):
        # Called when the model training starts
        pass
    
    def _on_step(self):
        # Called after each step in the environment
        if self.num_timesteps % self.eval_freq == 0:
            # Evaluate the current policy
            mean_reward, _ = evaluate_policy(
                self.model.policy,
                self.eval_env,
                n_eval_episodes=3,
                deterministic=True
            )
            
            # Update the best reward
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
            
            # Report to Optuna
            self.trial.report(self.best_mean_reward, self.num_timesteps)
            
            # Check if we should prune the trial
            if self.trial.should_prune():
                return False
        
        return True

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters using Optuna
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    
    # Use fixed values that work well with common batch sizes
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    
    # Using fewer environments for tuning speed
    n_envs = 2
    
    # Calculate if batch_size divides evenly into buffer_size
    buffer_size = n_steps * n_envs
    if buffer_size % batch_size != 0:
        print(f"Warning: batch_size {batch_size} does not divide evenly into buffer_size {buffer_size}")
        # We'll still proceed, but there will be a warning from SB3
    
    n_epochs = trial.suggest_int("n_epochs", 5, 30)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.999)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.01)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 0.9)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)
    
    # Create a specific log directory for this trial
    trial_log_dir = os.path.join(base_log_dir, f"trial_{trial.number}")
    os.makedirs(trial_log_dir, exist_ok=True)
    model_dir = os.path.join(trial_log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Log the suggested parameters
    with open(os.path.join(trial_log_dir, "params.txt"), "w") as f:
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"n_steps: {n_steps}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"n_epochs: {n_epochs}\n")
        f.write(f"gamma: {gamma}\n")
        f.write(f"gae_lambda: {gae_lambda}\n")
        f.write(f"clip_range: {clip_range}\n")
        f.write(f"ent_coef: {ent_coef}\n")
        f.write(f"vf_coef: {vf_coef}\n")
        f.write(f"max_grad_norm: {max_grad_norm}\n")
    
    # Create environment
    env, env_name = create_env(n_envs=n_envs)
    
    # Create evaluation environment
    eval_env, _ = create_env(n_envs=1)
    
    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best_model"),
        log_path=trial_log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=0
    )
    
    # Setup the proper Optuna callback
    optuna_callback = OptunaCallback(
        trial=trial,
        eval_env=eval_env,
        eval_freq=10000,
        verbose=0
    )
    
    # Create PPO model with suggested hyperparameters
    try:
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            tensorboard_log=trial_log_dir,
            device=device,
            verbose=0  # Set to 0 to reduce output during tuning
        )
        
        # Run a training session with reduced timesteps for faster tuning
        total_timesteps = 250000  # Reduced for tuning
        
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, optuna_callback],
            progress_bar=False  # Disable progress bar for cleaner output
        )
        
        # Final evaluation with more episodes
        mean_reward, std_reward = evaluate_policy(
            model.policy,
            eval_env,
            n_eval_episodes=10,
            deterministic=True
        )
        
        print(f"Trial {trial.number} finished: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
        
        # Save the final model
        model.save(os.path.join(model_dir, "final_model"))
        
        # Return the final mean reward as the objective value
        return mean_reward
    
    except (AssertionError, ValueError) as e:
        # Handle potential errors during training
        print(f"Trial {trial.number} failed with error: {e}")
        return float('-inf')
    finally:
        # Close environments
        env.close()
        eval_env.close()

def main():
    print("Starting Optuna hyperparameter optimization for PPO on AntMaze...")
    
    # Create a study object with pruning enabled
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=40000)
    study = optuna.create_study(
        direction="maximize", 
        pruner=pruner,
        study_name="sb3_ppo_antmaze_optimization"
    )
    
    # Run the optimization
    study.optimize(objective, n_trials=20, timeout=36000)  # 10 hours max, adjust as needed
    
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
    
    # Generate visualizations
    try:
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
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    # Train a final model with the best hyperparameters
    if study.best_params:
        print("\nTraining final model with best hyperparameters...")
        final_model_dir = os.path.join(base_log_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        
        # Create environments
        env, env_name = create_env(n_envs=4)  # Use full parallelization for final model
        eval_env, _ = create_env(n_envs=1)
        
        # Create model with best hyperparameters
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            tensorboard_log=final_model_dir,
            device=device,
            **study.best_params
        )
        
        # Setup eval callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=final_model_dir,
            log_path=final_model_dir,
            eval_freq=10000,
            deterministic=True,
            render=False
        )
        
        # Train for longer
        model.learn(total_timesteps=500000, callback=eval_callback)
        
        # Save the final model
        model.save(os.path.join(final_model_dir, "final_best_model"))
        
        # Save normalization stats for later use
        env.save(os.path.join(final_model_dir, "vec_normalize_stats"))
        
        # Final evaluation
        mean_reward, std_reward = evaluate_policy(
            model.policy,
            eval_env,
            n_eval_episodes=10,
            deterministic=True
        )
        
        print(f"Final model evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
        
        # Close environments
        env.close()
        eval_env.close()

if __name__ == "__main__":
    main()