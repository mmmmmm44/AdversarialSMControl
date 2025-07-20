import numpy as np
from stable_baselines3.common.callbacks import EveryNTimesteps, ConvertCallback, BaseCallback

from pathlib import Path
import json
from typing import Any

from model.H_network.h_network_rl_module import HNetworkRLModule
from model.H_network.h_network_arch import HNetworkType
from utils import print_log

class TrainHNetworkEveryNEpisodes(BaseCallback):
    """
    Callback that trains the H-network every n episodes during the RL training loop
    This callback is used to periodically train the H-network while the RL agent is being trained.
    """

    def __init__(self, every_n_episodes:int, h_network_rl_module: HNetworkRLModule):
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
    
    def __init__(self, every_n_episodes:int, h_network_rl_module: HNetworkRLModule, save_folder: Path):
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
        if self.save_folder is not None:
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
    
# TODO: change the log episode info to the env itself

# class LogEpisodeInfoEveryEpisode(BaseCallback):
#     """
#     Callback that logs episode information every episode during the RL training loop.
#     This callback is used to log information about the current episode, such as rewards and steps.
#     """

#     def __init__(self, target_folder:Path):
#         super().__init__()
#         self.n_episodes = 0
#         self.every_n_episodes = 1
#         self.last_episode_trigger = 0
#         self.target_folder = target_folder

#         assert self.target_folder is not None, "Target folder must be specified for logging episode info."

#     def _on_step(self) -> bool:
#         # Check that the `dones` local variable is defined
#         assert "dones" in self.locals, "`dones` variable is not defined, please check your code next to `callback.on_step()`"
#         self.n_episodes += np.sum(self.locals["dones"]).item()

#         if (self.n_episodes - self.last_episode_trigger) >= self.every_n_episodes:
#             self.last_episode_trigger = self.n_episodes
#             # Log the episode info
#             self._save_episode_info(self.locals, self.globals)
        
#         return True
    
#     def _save_episode_info(self, _locals: dict[str, Any], _globals: dict[str, Any]) -> bool:
#         """
#         Save the episode info list to a file.
#         Args:
#             _locals (dict[str, Any]): Local variables from the callback context.
#             _globals (dict[str, Any]): Global variables from the callback context.
#         """
#         env = self.model.get_env()

#         episode_info_list_vec = env.get_attr('episode_info_list', None)  # Ensure the environment has the episode_info_list attribute

#         if episode_info_list_vec is None:
#             print_log("\"episode_info_list\" attribute not found in the environment. Skipping saving.")
#             return False

#         # Create the folder if it does not exist
#         if not self.target_folder.exists():
#             self.target_folder.mkdir(parents=True)

#         # base on the self.locals["dones"] variable, we get the episode_info_list for a specific episode
#         dones = _locals["dones"]
#         assert len(episode_info_list_vec) == len(dones), "Mismatch between episode_info_list_vec and dones length. len(episode_info_list_vec): {}, len(dones): {}".format(len(episode_info_list_vec), len(dones))

#         _episode_start = self.n_episodes - np.sum(dones).item() + 1# temporary revert the number to get the correct episode number

#         for i in range(len(episode_info_list_vec)):
#             if not dones[i]:
#                 continue

#             episode_info_list = self.locals["infos"][i].get("episode_info_list")

#             if episode_info_list is None:
#                 print_log(f"Episode {_episode_start} info is None. Skipping saving.")
#             else:
#                 # Save the episode info to a file
#                 episode_info_path = self.target_folder / f"episode_{_episode_start:0>4}_info.json"
#                 with open(episode_info_path, 'w') as f:
#                     json.dump(episode_info_list, f, indent=4)

#                 print_log(f"Episode {_episode_start:0>4} info saved to {self.target_folder}")

#             _episode_start += 1  # Increment to the next episode number

#         return True