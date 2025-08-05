from decimal import Decimal
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from datetime import datetime
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent))
REPO_DIR = Path(__file__).parent

from utils import print_log
from utils_data_split import load_dataset, round_and_clip
from utils_logging import save_experiment_details
from rl_env.data_loader import SimpleSmartMeterDataLoader
from rl_env.training_mode import TrainingMode
from rl_env.base.env_module import SmartMeterEnvFactory, SmartMeterEnvironmentBase
from rl_env.env_callbacks import TrainHNetworkEveryNEpisodes, SaveCheckpointEveryNEpisodes, ValidateEveryNEpisodes, UpdateGlobalTimestepCallback, EnvLoggingCallback

from model.H_network.common.factories import create_h_network_module_with_defaults
from model.H_network.h_network_arch import HNetworkType

from model.DDQN.ddqn import DoubleDQN
from stable_baselines3.common.logger import configure


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main(training_kwargs: dict):

    reward_lambda = training_kwargs.get("reward_lambda", 0.5)
    action_type = training_kwargs.get("action_type", "continuous")
    n_episodes = training_kwargs.get("N_episodes", 600)  # total number of episodes to train
    aggregate_step_size = training_kwargs.get("aggregate_step_size", 50)  # default step size for aggregate load
    battery_step_size = training_kwargs.get("battery_step_size", Decimal("0.05"))  # default step size for battery charging action
    seed = training_kwargs.get("seed", None)

    rl_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = Path(REPO_DIR) / "experiments" / (rl_datetime + "_action_" + action_type + "_reward_lambda_" + str(reward_lambda))
    experiment_folder.mkdir(parents=True, exist_ok=True)

    # load datasets
    
    aggregate_load_segments_train, aggregate_load_df_train = load_dataset('train')
    aggregate_load_segments_validation, aggregate_load_df_validation = load_dataset('val')

    # quantize the aggregate load data to the nearest step size, then cap it to 5 kW
    # only applicable for discrete action type
    if action_type == "discrete":
        aggregate_load_df_train['aggregate'] = round_and_clip(aggregate_load_df_train['aggregate'], step_size=aggregate_step_size)
        aggregate_load_df_validation['aggregate'] = round_and_clip(aggregate_load_df_validation['aggregate'], step_size=aggregate_step_size)



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

    # --------------------
    # H-network parameters
    # --------------------

    h_network_kwargs = {
        "h_network_type": HNetworkType.H_NETWORK,
        "optimizer_class": torch.optim.RMSprop,
        "optimizer_kwargs": {"lr": 0.001},  # keep the same as the paper
    }

    # create h-network module
    h_network_rl_module = create_h_network_module_with_defaults(
        action_type=action_type,
        **h_network_kwargs,
        device=DEVICE
    )

    env_train = SmartMeterEnvFactory.create(
        action_type=action_type,
        smart_meter_data_loader=sm_dl_train,
        h_network_rl_module=h_network_rl_module,
        mode=TrainingMode.TRAIN,
        reward_lambda=reward_lambda,
        render_mode=None,

        # "continuous" action type does not require step size for battery charging and setting observation space size
        # "discrete" action type requires step size for battery charging and setting observation space size
        aggregate_step_size=aggregate_step_size,
        battery_step_size=battery_step_size,
    )

    env_validation = SmartMeterEnvFactory.create(
        action_type=action_type,
        smart_meter_data_loader=sm_dl_validation,
        h_network_rl_module=h_network_rl_module,
        mode=TrainingMode.VALIDATE, 
        reward_lambda=reward_lambda,
        render_mode=None,

        aggregate_step_size=aggregate_step_size,
        battery_step_size=battery_step_size,
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
    rl_lr = 0.00025
    rl_buffer_size = 150000  # scaled to match with our episode length
    rl_batch_size = 128     # keep the same as the paper
    rl_gamma = 1  # keep the same as the paper
    rl_train_freq = (120, "step")  # scaled to match with our episode length
    rl_target_update_interval = 7500  # scaled to match with our episode length
    policy_kwargs = {
        "optimizer_class": torch.optim.RMSprop,     # keep the same as the paper
    }

    rl_model = DoubleDQN(
        "MultiInputPolicy",
        env_train,
        learning_rate=rl_lr,      # keep the same as the paper
        buffer_size=rl_buffer_size,         # scaled to match with our episode length
        batch_size=rl_batch_size,             # keep the same as the paper
        gamma=rl_gamma,                       # keep the same as the paper
        train_freq=rl_train_freq,            # scaled to match with our episode length
        target_update_interval=rl_target_update_interval,    # scaled to match with our episode length
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=2,
    )
    rl_model.set_logger(logger)

    # also save the experiment details for future reference

    agent_parameters = {
        "learning_rate": rl_lr,
        "buffer_size": rl_buffer_size,
        "batch_size": rl_batch_size,
        "gamma": rl_gamma,
        "train_freq": rl_train_freq,
        "target_update_interval": rl_target_update_interval,
        "policy_kwargs": {
            "optimizer_class": policy_kwargs["optimizer_class"].__name__,
            # "optimizer_kwargs": policy_kwargs["optimizer_kwargs"]
        }
    }

    experiment_details = {
        "action_type": action_type,
        "agent_type": "DoubleDQN",
        "h_network_parameters": {
            "h_network_type": h_network_rl_module.h_network_type.name,
            "optimizer_class": h_network_rl_module.optimizer_class.__name__,
            "optimizer_kwargs": h_network_rl_module.optimizer_kwargs
        },
        "reward_lambda": env_train.reward_lambda,
        "dataloader_type": sm_dl_train.__class__.__name__,
        "reward_lambda": reward_lambda,
        "training_n_episodes": n_episodes,
        'agent_parameters': agent_parameters,
        'seed':seed
    }
    # additional details for discrete action type
    experiment_details = {**experiment_details, "aggregate_step_size": aggregate_step_size, "battery_step_size": float(battery_step_size)} if action_type == "discrete" else experiment_details

    # logging experiment details
    save_experiment_details(experiment_folder, experiment_details)
    env_train.save_environment_config(file_folder=experiment_folder)

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
        seed=seed
    )

    # for reproducibility, necessary calls to reset the h_network (create a new instance) and the environment
    h_network_rl_module.initialize_h_network_training()
    # env_train.reset(seed=seed); env_train.action_space.seed(seed)  # seed the action space for reproducibility
    env_validation.reset(seed=seed); env_validation.action_space.seed(seed)

    rl_model.learn(
        total_timesteps=24 * 60 * n_episodes,  # 600 episodes, each episode is 24 hours (with 1 min sample frequency)
        progress_bar=False,
        tb_log_name="DDQL_SmartMeterWorldDiscrete",
        callback=[
            UpdateGlobalTimestepCallback(),
            env_logging_callback,
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
        default="discrete",  # default to discrete action type
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
        "--aggregate_step_size",
        type=int,
        default=50,
        help="Step size for discretizing the aggregated user load (in W)."
    )
    parser.add_argument(
        "--battery_step_size",
        type=str,
        default="0.05",
        help="Step size for discretizing the battery charging action (in kW)."
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

    # Check step size equivalence (aggregate_step_size in W, battery_step_size in kW)
    if not np.isclose(args.aggregate_step_size / 1000.0, float(args.battery_step_size), atol=1e-6):
        parser.error("aggregate_step_size (in W) must be equivalent to battery_step_size (in kW).")

    training_kwargs = {
        "reward_lambda": args.reward_lambda,
        "action_type": args.action_type,
        "N_episodes": args.N_episodes,
        "aggregate_step_size": args.aggregate_step_size,
        "battery_step_size": Decimal(args.battery_step_size),
        "seed": args.seed
    }
    main(training_kwargs)
