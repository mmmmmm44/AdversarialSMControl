from typing import Any, Dict
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from datetime import datetime
import sys
import argparse
import json

sys.path.insert(0, str(Path(__file__).parent))
REPO_DIR = Path(__file__).parent

from utils import print_log
from utils_data_split import load_dataset
from utils_logging import save_experiment_details
from rl_env.data_loader import SimpleSmartMeterDataLoader
from rl_env.training_mode import TrainingMode
from rl_env.base.env_module import SmartMeterEnvFactory, SmartMeterEnvironmentBase
from rl_env.env_callbacks import TrainHNetworkEveryNEpisodes, SaveCheckpointEveryNEpisodes, ValidateEveryNEpisodes, EnvLoggingCallback, UpdateGlobalTimestepCallback

from model.H_network.common.factories import create_h_network_module_with_defaults
from model.H_network.h_network_arch import HNetworkType

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_PPO_params_to_dict(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert PPO parameters to a dictionary format for logging."""
    
    policy_kwargs = params.get("policy_kwargs", {})
    return {
        "learning_rate": params.get("learning_rate"),
        "batch_size": params.get("batch_size"),
        "n_steps": params.get("n_steps"),
        "n_epochs": params.get("n_epochs"),
        "gae_lambda": params.get("gae_lambda"),
        "policy_kwargs": {
            "net_arch": policy_kwargs.get("net_arch", []),
            "activation_fn": policy_kwargs.get("activation_fn", None).__name__ if policy_kwargs.get("activation_fn") else None,
        }
    }

def main(training_kwargs: dict):

    reward_lambda = training_kwargs.get("reward_lambda", 0.5)
    action_type = training_kwargs.get("action_type", "continuous")
    n_episodes = training_kwargs.get("N_episodes", 600)     # total number of episodes to train
    seed = training_kwargs.get("seed", None)                # random seed for reproducibility

    rl_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = Path(REPO_DIR) / "experiments" / (rl_datetime + "_action_" + action_type + "_reward_lambda_" + str(reward_lambda))
    experiment_folder.mkdir(parents=True, exist_ok=True)

    # load datasets
    
    aggregate_load_segments_train, aggregate_load_df_train = load_dataset('train')
    aggregate_load_segments_validation, aggregate_load_df_validation = load_dataset('val')


    # create data loaders
    sm_dl_train = SimpleSmartMeterDataLoader(
        aggregate_load_segments=aggregate_load_segments_train,
        aggregate_load_df=aggregate_load_df_train,
        registry_path=experiment_folder / "simple_episode_registry_train.json"
    )

    # create dataloader for validation
    sm_dl_validation = SimpleSmartMeterDataLoader(
        aggregate_load_segments=aggregate_load_segments_validation,
        aggregate_load_df=aggregate_load_df_validation,
        registry_path=experiment_folder / "simple_episode_registry_validation.json"
    )

    # create h-network module
    h_network_rl_module = create_h_network_module_with_defaults(
        action_type=action_type,
        h_network_type=HNetworkType.H_NETWORK,
        device=DEVICE
    )

    env_train_vec = make_vec_env(
        lambda: SmartMeterEnvFactory.create(
            action_type=action_type,
            smart_meter_data_loader=sm_dl_train,
            h_network_rl_module=h_network_rl_module,
            mode=TrainingMode.TRAIN,
            reward_lambda=reward_lambda,
            render_mode=None,
        ),
        seed=seed,
        n_envs=5,  # PPO works with vectorized environments (keep this same multiple of H-network training episodes)
        vec_env_cls=None,  # Use the default VecEnv class
    )

    env_validation = SmartMeterEnvFactory.create(
        action_type=action_type,
        smart_meter_data_loader=sm_dl_validation,
        h_network_rl_module=h_network_rl_module,
        mode=TrainingMode.VALIDATE, 
        reward_lambda=reward_lambda,
        render_mode=None,
    )
    

    h_network_rl_module.set_h_network(
        h_network_rl_module.initialize_h_network()
    )
    # h_network_rl_module.initialize_h_network_training()

    # all logs (json files, tensorboard) of training will be saved in the specified foler
    logger = configure(str(experiment_folder / "logs_tb"), ["stdout", "json", "tensorboard"])


    # --------------------
    # RL agent parameters
    # --------------------
    rl_lr = 7.3636e-3
    rl_batch_size = 64
    n_updates_btw_h_network_training = 2
    rl_gamma = 1            # same as the paper
    policy_kwargs = {
        "net_arch": [64, 64],               # same as the paper
        "activation_fn": torch.nn.ReLU,     # use ReLU instead of tanh (the default)
    }
    n_epochs = 15
    gae_lambda = 1 - 2.9011e-3


    # n_steps is automatically computed from n_updates_between_h_network_training
    # Target timesteps per environment per PPO update
    target_steps_per_env = 24 * 60 // n_updates_btw_h_network_training
    # Quantize n_steps to be a multiple of batch_size, close to target_steps_per_env
    n_steps = max(1, target_steps_per_env // rl_batch_size) * rl_batch_size
    # If n_steps is 0 (when target_steps_per_env < batch_size), set it to batch_size
    if n_steps == 0:
        n_steps = rl_batch_size

    agent_params = {
        "learning_rate": rl_lr,
        "n_steps": n_steps,
        "batch_size": rl_batch_size,
        "gamma": rl_gamma,
        "gae_lambda": gae_lambda,
        "n_epochs": n_epochs,
        "policy_kwargs": policy_kwargs,
    }

    rl_model = PPO(
        "MultiInputPolicy",
        env_train_vec,
        learning_rate=rl_lr,      # same as the paper
        n_steps=n_steps,          # same as the paper
        batch_size=rl_batch_size,             # same as the paper
        gamma=rl_gamma,                    # same as the paper
        gae_lambda=gae_lambda,              # same as the paper
        n_epochs=n_epochs,                  # same as the paper
        policy_kwargs=policy_kwargs,  # same as the paper
        seed=seed,  # set the seed for reproducibility, the seed will also be passed to the training environment.
        verbose=2,
    )

    # set the logger for the RL model
    rl_model.set_logger(logger)


    # also save the experiment details for future reference
    experiment_details = {
        "action_type": action_type,
        "agent_type": "PPO",
        "h_network_type": h_network_rl_module.h_network_type.name,
        "dataloader_type": sm_dl_train.__class__.__name__,
        "reward_lambda": env_train_vec.envs[0].env.reward_lambda,       # to get the true instance of the environment
        "training_n_episodes": n_episodes,
        'agent_parameters': convert_PPO_params_to_dict(agent_params),
        "seed": seed,
    }
    # additional details for discrete action type

    # logging experiment details
    save_experiment_details(experiment_folder, experiment_details)
    env_train_vec.envs[0].env.save_environment_config(experiment_folder)

    # train
    every_n_episodes = 5
    sweep_every_n_episodes = 5 * 10  # every 50 episodes

    env_logging_callback = EnvLoggingCallback(log_folder=experiment_folder / "logs_train")

    eval_callback = ValidateEveryNEpisodes(
        every_n_episodes=every_n_episodes * 5,
        sweep_every_n_episodes=sweep_every_n_episodes,
        validation_log_folder=experiment_folder / "logs_validation",
        validation_env=env_validation,
        h_network_rl_module=h_network_rl_module,
        best_model_save_path=experiment_folder / "best_model",
        enable_sweep=True,
        seed=seed
    )

    # for reproducibility, necessary calls to reset the h_network (create a new instance) and the environment
    h_network_rl_module.initialize_h_network_training()
    # env_train_vec.reset()   # commenting this as the reset is done in rl_model
    env_validation.reset(seed=seed); env_validation.action_space.seed(seed)  # seed the action space for reproducibility

    rl_model.learn(
        total_timesteps=24 * 60 * n_episodes,  # 600 episodes, each episode is 24 hours (with 1 min sample frequency)
        progress_bar=False,
        tb_log_name="PPO_SmartMeterWorldContinuous",
        callback=[
            UpdateGlobalTimestepCallback(),
            env_logging_callback,
            # Train the H-network every n episodes
            TrainHNetworkEveryNEpisodes(every_n_episodes=every_n_episodes, h_network_rl_module=h_network_rl_module),
            SaveCheckpointEveryNEpisodes(every_n_episodes=every_n_episodes, h_network_rl_module=h_network_rl_module, save_folder=experiment_folder / "checkpoints"),
            eval_callback
        ]
    )

    # save the training results
    print_log("Saving training results...")

    SmartMeterEnvironmentBase.save_episodes_rewards(env_logging_callback.get_episode_rewards(), experiment_folder / "logs_train", TrainingMode.TRAIN)
    h_network_rl_module.save_train_loss_list(experiment_folder / "logs_train" / "h_network_train_loss_list.json")

    # save both RL model and H-network at the end of training
    rl_model_path = experiment_folder / f"rl_model.zip"
    h_network_path = experiment_folder / f"h_network2.pth" if h_network_rl_module.h_network_type == HNetworkType.H_NETWORK2 else experiment_folder / f"h_network.pth"

    rl_model.save(rl_model_path)
    h_network_rl_module.save_h_network(h_network_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train RL agent for Smart Meter environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--action_type",
        choices=["continuous"],
        default="continuous",
        help="Type of RL action space: 'continuous' (PPO) or 'discrete' (DoubleDQN)."
    )
    parser.add_argument(
        "--reward_lambda",
        type=float,
        default=0.5,
        help="Reward lambda value (trade-off parameter), must be between 0 and 1 inclusive."
    )
    parser.add_argument(
        "--N_episodes",
        type=int,
        default=600,
        help="Number of training episodes (int >= 50)."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility of environment episode sampling, RL agent, and H-network initialization. This will be used to seed the action space and the np_random in the environment for both training env and validation env. Also, while creating the rl_model, passing the seed to rl_model initialization will call stable_baselines3.common.utils.set_random_seed(seed) to set the random seed for python RNG, np.random.seed(seed), and torch.manual_seed(seed). (OS: this should greatly ensure the reproducibility, but not guaranteed, as torch.manual_seed(seed) does not guarantee the same results across different PyTorch versions or hardware/platform configurations.)"
    )

    args = parser.parse_args()

    # Type and value checks
    if args.N_episodes < 50:
        parser.error("N_episodes must be >= 50.")
    if not (0.0 <= args.reward_lambda <= 1.0):
        parser.error("reward_lambda must be between 0 and 1 inclusive.")

    training_kwargs = {
        "reward_lambda": args.reward_lambda,
        "action_type": args.action_type,
        "N_episodes": args.N_episodes,
        "seed": args.seed
    }
    main(training_kwargs)
