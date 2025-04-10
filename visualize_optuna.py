import os
import optuna
import plotly

# First, make sure plotly is installed
try:
    import plotly
except ImportError:
    print("Installing plotly...")
    import subprocess
    subprocess.check_call(["pip", "install", "plotly"])
    import plotly

import optuna.visualization as vis

def visualize_study(study_name="ppo_antmaze_optimization", storage_path=None, output_dir=None):
    """
    Load an existing Optuna study and generate visualizations
    
    Parameters:
    -----------
    study_name : str
        Name of the study to visualize
    storage_path : str
        Path to the storage (either file path or SQLite URL)
    output_dir : str
        Directory to save visualizations
    """
    # Set default storage path if not provided
    if storage_path is None:
        # Try different common locations
        possible_paths = [
            f"sqlite:///{study_name}.db",                   # Current directory
            f"sqlite:///logs/{study_name}.db",              # logs subdirectory
            f"sqlite:///logs/optuna_{study_name}.db",       # logs with prefix
        ]
        
        for path in possible_paths:
            try:
                print(f"Trying to load study from {path}...")
                study = optuna.load_study(study_name=study_name, storage=path)
                print(f"Successfully loaded study from {path}")
                break
            except Exception as e:
                print(f"Could not load from {path}: {e}")
        else:
            # If no storage worked, ask for manual path
            print("\nCould not find study storage automatically.")
            print("Please specify the path to your database or provide a storage URL.")
            print("For example: sqlite:///my_study.db")
            storage_input = input("Storage path: ")
            
            try:
                study = optuna.load_study(study_name=study_name, storage=storage_input)
            except Exception as e:
                print(f"Error loading study: {e}")
                return
    else:
        # Use the provided storage path
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_path)
        except Exception as e:
            print(f"Error loading study from {storage_path}: {e}")
            return
    
    # Create output directory if not specified
    if output_dir is None:
        output_dir = f"logs/optuna_visualization_{study_name}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Print study information
    print(f"\nSuccessfully loaded study '{study_name}' with {len(study.trials)} trials.")
    
    if len(study.trials) == 0:
        print("No trials found in the study. Nothing to visualize.")
        return
    
    if study.best_trial is not None:
        print(f"Best value: {study.best_value:.3f} from trial {study.best_trial.number}")
        print("\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
    else:
        print("No completed trials with values.")
        return
    
    # Generate and save visualizations
    print("\nGenerating visualizations...")
    
    # Plot optimization history
    try:
        fig = vis.plot_optimization_history(study)
        output_file = os.path.join(output_dir, "optimization_history.html")
        fig.write_html(output_file)
        print(f"Saved optimization history to {output_file}")
    except Exception as e:
        print(f"Error generating optimization history: {e}")
    
    # Plot parameter importances
    try:
        fig = vis.plot_param_importances(study)
        output_file = os.path.join(output_dir, "param_importances.html")
        fig.write_html(output_file)
        print(f"Saved parameter importances to {output_file}")
    except Exception as e:
        print(f"Error generating parameter importances: {e}")
    
    # Plot parameter relationships
    try:
        fig = vis.plot_parallel_coordinate(study)
        output_file = os.path.join(output_dir, "parallel_coordinate.html")
        fig.write_html(output_file)
        print(f"Saved parallel coordinate plot to {output_file}")
    except Exception as e:
        print(f"Error generating parallel coordinate plot: {e}")
    
    # Plot slice plot
    try:
        fig = vis.plot_slice(study)
        output_file = os.path.join(output_dir, "slice_plot.html")
        fig.write_html(output_file)
        print(f"Saved slice plot to {output_file}")
    except Exception as e:
        print(f"Error generating slice plot: {e}")
    
    # Plot contour plot if the study has at least 2 parameters
    if len(study.best_params) >= 2:
        try:
            fig = vis.plot_contour(study)
            output_file = os.path.join(output_dir, "contour_plot.html")
            fig.write_html(output_file)
            print(f"Saved contour plot to {output_file}")
        except Exception as e:
            print(f"Error generating contour plot: {e}")
    
    print(f"\nAll visualizations saved to {output_dir}")

if __name__ == "__main__":
    # You can directly use the sqlite file if you know its location
    visualize_study(
        study_name="ppo_antmaze_optimization", 
        storage_path="sqlite:///ppo_antmaze_optimization.db"
    )
    
    # Alternatively, you can let the script try to find it automatically:
    # visualize_study(study_name="ppo_antmaze_optimization")