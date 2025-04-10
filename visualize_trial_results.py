import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def visualize_results_from_logs(base_dir=None):
    """
    Visualize results from Optuna trials folder structure
    """
    # Find the optuna_tuning folder if not specified
    if base_dir is None:
        log_dirs = glob.glob("logs/optuna_tuning_*")
        if not log_dirs:
            print("No optuna_tuning folders found.")
            return
        base_dir = max(log_dirs, key=os.path.getmtime)  # Get the most recent
    
    print(f"Analyzing results from: {base_dir}")
    
    # Find all trial folders
    trial_dirs = glob.glob(os.path.join(base_dir, "trial_*"))
    if not trial_dirs:
        print("No trial folders found.")
        return
    
    # Collect results from each trial
    results = []
    for trial_dir in trial_dirs:
        trial_num = int(os.path.basename(trial_dir).split("_")[1])
        
        # Get parameters
        params_file = os.path.join(trial_dir, "params.txt")
        params = {}
        if os.path.exists(params_file):
            with open(params_file, "r") as f:
                for line in f:
                    if ":" in line:
                        key, value = line.strip().split(":", 1)
                        try:
                            params[key.strip()] = float(value.strip())
                        except ValueError:
                            params[key.strip()] = value.strip()
        
        # Get rewards
        rewards_file = os.path.join(trial_dir, "rewards.csv")
        final_reward = None
        if os.path.exists(rewards_file):
            try:
                rewards_df = pd.read_csv(rewards_file)
                if not rewards_df.empty:
                    final_reward = rewards_df["avg_reward"].iloc[-1]
            except Exception as e:
                print(f"Error reading rewards for trial {trial_num}: {e}")
        
        if final_reward is not None:
            # Store trial results
            results.append({
                "trial_num": trial_num,
                "reward": final_reward,
                "params": params,
                "rewards_file": rewards_file if os.path.exists(rewards_file) else None
            })
    
    if not results:
        print("No valid results found in trials.")
        return
    
    # Sort by performance
    results.sort(key=lambda x: x["reward"], reverse=True)
    
    # Display best results
    print("\n===== RESULTS SUMMARY =====")
    print(f"Found {len(results)} completed trials.")
    print("\nTop 5 Performing Trials:")
    
    for i, result in enumerate(results[:5]):
        print(f"\n{i+1}. Trial {result['trial_num']}: Reward = {result['reward']:.4f}")
        for key, value in result['params'].items():
            print(f"   {key}: {value}")
    
    # Plot learning curves of top trials
    plt.figure(figsize=(12, 6))
    
    for i, result in enumerate(results[:5]):
        if result["rewards_file"]:
            rewards_df = pd.read_csv(result["rewards_file"])
            plt.plot(rewards_df["episode"], rewards_df["avg_reward"], 
                     label=f"Trial {result['trial_num']} (Final: {result['reward']:.4f})")
    
    plt.title("Learning Curves of Top Performing Trials")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    output_dir = os.path.join(base_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "top_learning_curves.png"))
    
    # Save the best parameters to a file
    with open(os.path.join(output_dir, "best_params.txt"), "w") as f:
        f.write(f"Best Trial: {results[0]['trial_num']}\n")
        f.write(f"Reward: {results[0]['reward']:.4f}\n\n")
        f.write("Parameters:\n")
        for key, value in results[0]['params'].items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nAnalysis saved to {output_dir}")
    print(f"Best parameters saved to {os.path.join(output_dir, 'best_params.txt')}")
    
    # Return the best parameters for easy access
    return results[0]['params']

if __name__ == "__main__":
    best_params = visualize_results_from_logs()
    print("\nBest hyperparameters for training:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")