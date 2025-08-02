from decimal import Decimal
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
from utils_data_split import load_dataset, round_and_clip
from utils_logging import save_experiment_details
from rl_env.data_loader import SimpleSmartMeterDataLoader
from rl_env.training_mode import TrainingMode
from rl_env.base.env_module import SmartMeterEnvFactory
from rl_env.env_callbacks import TrainHNetworkEveryNEpisodes, SaveCheckpointEveryNEpisodes, ValidateEveryNEpisodes

from model.H_network.common.factories import create_h_network_module_with_defaults
from model.H_network.h_network_arch import HNetworkType

from stable_baselines3 import PPO
from model.DDQN.ddqn import DoubleDQN
from stable_baselines3.common.logger import configure


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main(training_kwargs: dict):

    reward_lambda = training_kwargs.get("reward_lambda", 0.5)
    action_type = training_kwargs.get("action_type", "continuous")
    n_episodes = training_kwargs.get("N_episodes", 600)  # total number of episodes to train

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

    env_train = SmartMeterEnvFactory.create(
        action_type=action_type,
        smart_meter_data_loader=sm_dl_train,
        h_network_rl_module=h_network_rl_module,
        mode=TrainingMode.TRAIN,
        log_folder=experiment_folder / 'logs_train',
        reward_lambda=reward_lambda,
        # render_mode="human",
        render_mode=None,
    )

    env_validation = SmartMeterEnvFactory.create(
        action_type=action_type,
        smart_meter_data_loader=sm_dl_validation,
        h_network_rl_module=h_network_rl_module,
        mode=TrainingMode.VALIDATE, 
        log_folder=experiment_folder / 'logs_validation',
        reward_lambda=reward_lambda,
        # render_mode="human",
        render_mode=None,
    )

    h_network_rl_module.set_h_network(
        h_network_rl_module.initialize_h_network()
    )
    h_network_rl_module.initialize_h_network_training()

    # all logs (json files, tensorboard) of training will be saved in the specified foler
    logger = configure(str(experiment_folder / "logs_train"), ["stdout", "json", "tensorboard"])


    # --------------------
    # RL agent parameters
    # --------------------
    rl_lr = 2.5e-4          # lets use the same learning rate as the paper
    rl_batch_size=128       # same as the paper
    rl_gamma = 1            # same as the paper

    rl_model = PPO(
        "MultiInputPolicy", 
        env_train,
        learning_rate=rl_lr,
        batch_size=rl_batch_size,
        gamma=rl_gamma,
        verbose=2,
    )


    rl_model.set_logger(logger)

    agent_parameters = {
        "learning_rate": rl_lr,
        "batch_size": rl_batch_size,
        "gamma": rl_gamma,
    }


    # also save the experiment details for future reference
    experiment_details = {
        "action_type": action_type,
        "agent_type": "PPO",
        "h_network_type": h_network_rl_module.h_network_type.name,
        "reward_lambda": env_train.reward_lambda,
        "dataloader_type": sm_dl_train.__class__.__name__,
        "reward_lambda": reward_lambda,
        "training_n_episodes": n_episodes,
        'agent_parameters': agent_parameters,
    }
    # additional details for discrete action type

    # logging experiment details
    save_experiment_details(experiment_folder, experiment_details)
    env_train.save_environment_config()

    # train
    every_n_episodes = 5
    sweep_every_n_episodes = 5 * 10  # every 50 episodes

    rl_model.learn(
        total_timesteps=24 * 60 * n_episodes,  # 600 episodes, each episode is 24 hours (with 1 min sample frequency)
        progress_bar=False,
        tb_log_name="PPO_SmartMeterWorldContinuous",
        callback=[
            TrainHNetworkEveryNEpisodes(every_n_episodes=every_n_episodes, h_network_rl_module=h_network_rl_module),
            SaveCheckpointEveryNEpisodes(every_n_episodes=every_n_episodes, h_network_rl_module=h_network_rl_module, save_folder=experiment_folder / "checkpoints"),
            ValidateEveryNEpisodes(every_n_episodes=every_n_episodes * 5, sweep_every_n_episodes=sweep_every_n_episodes, validation_log_folder=experiment_folder / "logs_validation", validation_env=env_validation, h_network_rl_module=h_network_rl_module, best_model_save_path=experiment_folder / "best_model")
        ]
    )

    # save the training results
    print_log("Saving training results...")

    env_train.save_episodes_rewards(experiment_folder / "logs_train")
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
        choices=["continuous", "discrete"],
        required=True,
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

    args = parser.parse_args()

    # Type and value checks
    if args.N_episodes < 50:
        parser.error("N_episodes must be >= 50.")
    if not (0.0 <= args.reward_lambda <= 1.0):
        parser.error("reward_lambda must be between 0 and 1 inclusive.")

    # Check step size equivalence (aggregate_step_size in W, battery_step_size in kW)
    if not np.isclose(args.aggregate_step_size / 1000.0, float(args.battery_step_size), atol=1e-6):
        parser.error("aggregate_step_size (in W) must be equivalent to battery_step_size (in kW).")

    training_kwargs = {
        "reward_lambda": args.reward_lambda,
        "action_type": args.action_type,
        "N_episodes": args.N_episodes,
    }
    main(training_kwargs)
