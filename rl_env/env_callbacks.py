import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EventCallback

from pathlib import Path
import json
from typing import Any
from datetime import datetime

from model.H_network.base.h_network_module import HNetworkRLModuleBase
from model.H_network.h_network_arch import HNetworkType
from utils import print_log

class TrainHNetworkEveryNEpisodes(BaseCallback):
    """
    Callback that trains the H-network every n episodes during the RL training loop
    This callback is used to periodically train the H-network while the RL agent is being trained.
    """

    def __init__(self, every_n_episodes:int, h_network_rl_module: HNetworkRLModuleBase):
        super().__init__()

        if every_n_episodes <= 0:
            raise ValueError("every_n_episodes must be a positive integer.")

        self.n_episodes = 0
        self.every_n_episodes = every_n_episodes
        self.last_episode_trigger = 0
        self.h_network_rl_module = h_network_rl_module


    def _on_step(self) -> bool:
        # Check that the `dones` local variable is defined
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
        self.n_episodes += np.sum(self.locals["dones"]).item()

        if (self.n_episodes - self.last_episode_trigger) >= self.every_n_episodes:
            self.last_episode_trigger = self.n_episodes

            return self._train(self.locals, self.globals)

        return True

    def _train(self, _locals: dict[str, Any], _globals: dict[str, Any]) -> bool:
        
        print_log("Training H-network...")

        # Train the H-network
        self.h_network_rl_module.train()

        if self.h_network_rl_module.train_loss_list is None:
            print_log("No training loss data available. Skipping logging.")
            return True

        # Log the training loss
        latest_loss_item = self.h_network_rl_module.train_loss_list[-1]
        mean_loss = latest_loss_item['mean_loss']
        std_loss = latest_loss_item['std_loss']
        num_batches = latest_loss_item['num_batches']
        print_log(f"H-network training (mean, std, num_batches): ({mean_loss:.8f}, {std_loss:.8f}, {num_batches})")

        return True
    
class SaveCheckpointEveryNEpisodes(BaseCallback):
    """
    Callback that saves the H-network and RL agent models every n episodes during the RL training loop.
    This callback is used to periodically save the models to a specified folder.
    """
    
    def __init__(self, every_n_episodes:int, h_network_rl_module: HNetworkRLModuleBase, save_folder: Path):
        super().__init__()

        if every_n_episodes <= 0:
            raise ValueError("every_n_episodes must be a positive integer.")

        self.n_episodes = 0
        self.every_n_episodes = every_n_episodes
        self.last_episode_trigger = 0
        self.h_network_rl_module = h_network_rl_module
        self.save_folder = save_folder


    def _init_callback(self) -> None:
        # Create folder if needed
        if ((self.save_folder is not None) and (not self.save_folder.exists())):
            Path(self.save_folder).mkdir(parents=True)

    def _on_step(self) -> bool:
        # Check that the `dones` local variable is defined
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
        self.n_episodes += np.sum(self.locals["dones"]).item()

        if (self.n_episodes - self.last_episode_trigger) >= self.every_n_episodes:
            self.last_episode_trigger = self.n_episodes
            self._save_checkpoint(self.locals, self.globals)

        return True
    
    def _save_checkpoint(self, _locals: dict[str, Any], _globals: dict[str, Any]) -> bool:
        """
        Save the H-network and RL agent models to the specified folder.
        """
        if self.save_folder is None:
            print_log("No save folder specified. Skipping saving.")
            return True

        # create a folder for each saved episode
        episode_save_folder = self.save_folder / f"episode_{self.n_episodes:0>4}"
        if not episode_save_folder.exists():
            episode_save_folder.mkdir(parents=True)

        # Save the H-network
        h_network_path = episode_save_folder / f"{'h_network2' if self.h_network_rl_module.h_network_type.value == HNetworkType.H_NETWORK2.value else 'h_network'}_{self.n_episodes:0>4}.pth"
        self.h_network_rl_module.save_h_network(h_network_path)

        # Save the RL model
        rl_model_path = episode_save_folder / f"rl_model_{self.n_episodes:0>4}.zip"
        self.model.save(rl_model_path)

        print_log(f"Models saved at {h_network_path} and {rl_model_path}")
        return True
    

class ValidateEveryNEpisodes(EventCallback):
    """
    Callback that performs validation every N episodes during the RL training loop.
    This callback evaluates the model on a validation environment using either quick sampling
    or complete sweep based on the validation frequency. It also tracks and saves the best models.
    """

    def __init__(self, every_n_episodes: int, sweep_every_n_episodes: int, validation_log_folder: Path, validation_env, h_network_rl_module: HNetworkRLModuleBase, best_model_save_path: Path):
        super().__init__()
        self.n_episodes = 0
        self.every_n_episodes = every_n_episodes
        self.sweep_every_n_episodes = sweep_every_n_episodes
        self.last_episode_trigger = 0

        assert every_n_episodes > 0, "every_n_episodes must be a positive integer."
        assert sweep_every_n_episodes > 0, "sweep_every_n_episodes must be a positive integer."
        assert sweep_every_n_episodes % every_n_episodes == 0, "sweep_every_n_episodes must be a multiple of every_n_episodes."

        self.validation_log_folder = validation_log_folder
        self.validation_env = validation_env
        self.h_network_rl_module = h_network_rl_module
        
        # Best model tracking
        self.best_mean_reward = -np.inf
        self.best_episode = None
        self.best_model_save_path = best_model_save_path

        assert self.validation_log_folder is not None, "Validation log folder must be specified for logging validation results."
        assert best_model_save_path is not None and isinstance(best_model_save_path, Path), "best_model_save_path must be a Path object."

        if not self.validation_log_folder.exists():
            self.validation_log_folder.mkdir(parents=True)

        # Create best model save folder if specified
        if not self.best_model_save_path.exists():
            self.best_model_save_path.mkdir(parents=True)

    def _on_step(self) -> bool:
        # Check that the `dones` local variable is defined
        assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
        self.n_episodes += np.sum(self.locals["dones"]).item()

        if (self.n_episodes - self.last_episode_trigger) >= self.every_n_episodes:
            self.last_episode_trigger = self.n_episodes
            # Log the episode info
            self._validate(self.locals, self.globals)

        return True
    
    def _validate(self, _locals: dict[str, Any], _globals: dict[str, Any]) -> bool:
        """
        Custom validation function that performs either quick random sampling or complete sweep
        based on validation frequency. Quick sampling (5-10 episodes) for regular H-network updates,
        complete sweep (all 73 episodes) for critical improvements every ~50 training episodes.
        
        Args:
            _locals: Local variables from the training loop
            _globals: Global variables from the training loop
            
        Returns:
            bool: True to continue training, False to stop
        """
        
        # Get the current RL model from the training loop
        model = self.model
        
        # Get validation environment
        if self.validation_env is None:
            print_log("No validation environment provided. Skipping validation.")
            return True
        
        # alter the log_folder to include the current episode number
        self.validation_env.set_log_folder(self.validation_log_folder / f"episode_{self.n_episodes:0>4}")      # it will create the folder if it does not exist
        
        total_episodes = self.validation_env.smart_meter_data_loader.get_divided_segments_length()
        
        # Determine validation strategy based on episode count
        # Complete sweep every 50 episodes (10 H-network updates), quick sampling otherwise
        is_critical_validation = (self.n_episodes % 50) == 0 and self.n_episodes > 0
        
        if is_critical_validation:
            # Complete sweep for critical validations
            print_log(f"Starting CRITICAL validation - complete sweep on all {total_episodes} episodes...")
            episode_indices = list(range(total_episodes))
            validation_type = "critical"
        else:
            # Quick random sampling for regular validations
            sample_size = min(10, total_episodes)  # Sample 10 episodes or all if fewer available
            episode_indices = np.random.choice(total_episodes, size=sample_size, replace=False).tolist()
            print_log(f"Starting QUICK validation - random sampling on {sample_size} episodes...")
            validation_type = "quick"
        
        # Initialize containers for collecting results
        all_episode_rewards = []
        all_episode_lengths = []
        all_episode_infos = []
        
        # Evaluate on selected episodes
        for i, episode_idx in enumerate(episode_indices):
            # Reset environment to specific episode
            obs, info = self.validation_env.reset(episode_idx=episode_idx)
            
            # Initialize episode tracking variables
            episode_reward = 0.0
            episode_length = 0
            done = False
            state = None  # For recurrent policies
            
            # Run the episode until completion
            while not done:
                # Get action from the model (deterministic evaluation)
                action, state = model.predict(obs, state=state, deterministic=True)
                
                # Take step in environment
                obs, reward, done, truncated, info = self.validation_env.step(action)
                
                # Accumulate episode statistics
                episode_reward += reward
                episode_length += 1
            
            # Store episode results
            all_episode_rewards.append(episode_reward)
            all_episode_lengths.append(episode_length)
            all_episode_infos.append({
                'episode_idx': episode_idx,
                'episode_info': self.validation_env.episode.get_episode_info() if hasattr(self.validation_env, 'episode') else {}
            })
            
            # Log progress for critical validations
            if is_critical_validation and (i + 1) % 10 == 0:
                print_log(f"Critical validation progress: {i + 1}/{len(episode_indices)} episodes completed")
        
        # Calculate comprehensive statistics
        mean_reward = np.mean(all_episode_rewards)
        std_reward = np.std(all_episode_rewards)
        min_reward = np.min(all_episode_rewards)
        max_reward = np.max(all_episode_rewards)
        median_reward = np.median(all_episode_rewards)
        
        mean_length = np.mean(all_episode_lengths)
        std_length = np.std(all_episode_lengths)
        
        # Log results based on validation type
        if is_critical_validation:
            print_log(f"CRITICAL Validation Results (N={len(episode_indices)} episodes):")
            print_log(f"  Reward - Mean: {mean_reward:.4f}, Std: {std_reward:.4f}")
            print_log(f"  Reward - Min: {min_reward:.4f}, Max: {max_reward:.4f}, Median: {median_reward:.4f}")
            print_log(f"  Episode Length - Mean: {mean_length:.2f}, Std: {std_length:.2f}")
        else:
            print_log(f"Quick Validation Results (N={len(episode_indices)} episodes):")
            print_log(f"  Reward - Mean: {mean_reward:.4f} ± {std_reward:.4f}")
            print_log(f"  Episode Length - Mean: {mean_length:.2f}")
        
        # Prepare validation data
        validation_data = {
            'episode': self.n_episodes,
            'timestamp': self.model.num_timesteps,
            'validation_type': validation_type,
            'evaluated_episodes': len(episode_indices),
            'total_available_episodes': total_episodes,
            'episode_indices': episode_indices,
            'episode_rewards': all_episode_rewards,
            'episode_lengths': all_episode_lengths,
            'episode_infos': all_episode_infos,
            'summary_stats': {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'min_reward': min_reward,
                'max_reward': max_reward,
                'median_reward': median_reward,
                'mean_length': mean_length,
                'std_length': std_length
            }
        }
        
        # Save validation results
        validation_file = self.validation_log_folder / f"validation_{validation_type}_episode_{self.n_episodes:0>4}.json"
        with open(validation_file, 'w') as f:
            # Convert numpy types to Python native types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                return obj
            
            json.dump(convert_numpy(validation_data), f, indent=2, default=str)
        
        print_log(f"Validation results saved to {validation_file}")
        
        # Track best performance and save best models for critical validations
        if is_critical_validation and self.best_model_save_path:
            if mean_reward > self.best_mean_reward:
                print_log(f"New best validation reward: {mean_reward:.4f} (previous: {self.best_mean_reward:.4f})")
                
                # Update best performance tracking
                self.best_mean_reward = mean_reward
                self.best_episode = self.n_episodes
                
                # Save best models
                best_rl_path = self.best_model_save_path / "best_rl_model.zip"
                best_h_network_path = self.best_model_save_path / f"best_{'h_network2' if self.h_network_rl_module.h_network_type.value == HNetworkType.H_NETWORK2.value else 'h_network'}.pth"
                
                model.save(best_rl_path)
                self.h_network_rl_module.save_h_network(best_h_network_path)
                
                # Save metadata
                best_model_info = {
                    "best_episode": self.best_episode,
                    "best_mean_reward": float(self.best_mean_reward),
                    "best_std_reward": float(std_reward),
                    "validation_type": validation_type,
                    "evaluated_episodes": len(episode_indices),
                    "timestamp": self.model.num_timesteps,
                    "saved_at": datetime.now().isoformat(),
                    "model_paths": {
                        "rl_model": "best_rl_model.zip",
                        "h_network": f"best_{'h_network2' if self.h_network_rl_module.h_network_type.value == HNetworkType.H_NETWORK2.value else 'h_network'}.pth"
                    }
                }
                
                with open(self.best_model_save_path / "best_model_info.json", 'w') as f:
                    json.dump(best_model_info, f, indent=2)
                
                print_log(f"Best models saved to {self.best_model_save_path}")
        
        return True