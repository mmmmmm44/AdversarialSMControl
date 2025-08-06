from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from datetime import datetime
import sys

import gymnasium
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

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

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_HYPERPARAMS = {
    "policy": "MultiInputPolicy",
    "verbose": 2,
}

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters."""

    learning_rate = trial.suggest_float("PPO_learning_rate", 1e-5, 0.1, log=True)
    batch_size = trial.suggest_categorical("PPO_batch_size", [64, 128, 256])
    n_updates_btw_h_network_training = trial.suggest_int("PPO_n_updates_btw_h_network_training", 2, 8)

    n_envs = 5  # Fixed - n_envs should be a factor of episodes per H-network update `train_hnetwork_every_n_episodes`
    
    # Calculate n_steps such that:
    # 1. Each PPO update sees (24 * 60 / n_updates_btw_h_network_training) timesteps per environment
    # 2. n_steps * n_envs must be divisible by batch_size
    
    # Target timesteps per environment per PPO update
    target_steps_per_env = 24 * 60 // n_updates_btw_h_network_training
    
    # Quantize n_steps to be a multiple of batch_size, close to target_steps_per_env
    n_steps = max(1, target_steps_per_env // batch_size) * batch_size
    
    # If n_steps is 0 (when target_steps_per_env < batch_size), set it to batch_size
    if n_steps == 0:
        n_steps = batch_size

    n_epochs = trial.suggest_int("PPO_n_epochs", 1, 20)
    gae_lambda = 1.0 - trial.suggest_float("PPO_gae_lambda", 0.001, 0.2, log=True)

    # also sample the activation function
    activation_fn = trial.suggest_categorical("PPO_activation_fn", ["tanh", "relu"])
    if activation_fn == "relu":
        activation_fn = nn.ReLU
    else:
        activation_fn = nn.Tanh

    policy_kwargs = {
        "net_arch": [64, 64],
        "activation_fn": activation_fn,
    }

    # Store computed values and metadata for analysis
    trial.set_user_attr("PPO_n_steps", n_steps)
    trial.set_user_attr("n_envs", n_envs)
    trial.set_user_attr("PPO_buffer_size", n_steps * n_envs)
    trial.set_user_attr("target_steps_per_env_per_PPO_update", target_steps_per_env)
    # trial.set_user_attr("actual_steps_per_env_per_PPO_update", n_steps)

    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "policy_kwargs": policy_kwargs,
    }

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

def sample_h_network_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for H-Network hyperparameters."""

    h_network_lr = trial.suggest_float("HNetwork_lr", 1e-4, 0.01, log=True)

    return {
        "optimizer_kwargs": {'lr': h_network_lr},
    }



def main(training_kwargs: dict, study:optuna.Study):

    reward_lambda = training_kwargs.get("reward_lambda", 0.5)
    action_type = training_kwargs.get("action_type", "continuous")
    n_episodes = training_kwargs.get("N_episodes", 50)  # total number of episodes to train
    seed = training_kwargs.get("seed", 42)  # for reproducibility

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


    def objective(trial: optuna.Trial) -> float:
        """Objective function for the Optuna study."""

        trial_number = trial.number
        print_log(f"Starting trial {trial_number:04d} with parameters: {trial.params}")

        trial_folder = experiment_folder / f"trial_{trial_number:04d}"
        trial_folder.mkdir(parents=True, exist_ok=True)
        
        # n_envs is fixed at 5
        n_envs = 5
        
        # Sample hyperparameters
        params = sample_ppo_params(trial)
        h_network_params = sample_h_network_params(trial)
        
        # create h-network module
        h_network_rl_module = create_h_network_module_with_defaults(
            action_type='continuous',
            h_network_type=HNetworkType.H_NETWORK,
            device=DEVICE,
            **h_network_params
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
            n_envs=n_envs,
            seed=seed,
            vec_env_cls=None,  # Use default vectorized environment
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
        logger = configure(str(trial_folder / "logs_tb"), ["stdout", "json", "tensorboard"])


        # --------------------
        # RL agent parameters
        # --------------------

        kwargs = DEFAULT_HYPERPARAMS.copy()
        kwargs.update(params)
        kwargs.update({
            "seed": seed
        })

        rl_model = PPO(
            env=env_train_vec,      # this automatically calls env_train_vec.reset(), which resets every environment in the vectorized environment, and their action spaces.
            **kwargs
        )

        rl_model.set_logger(logger)

        # also save the experiment details for future reference
        experiment_details = {
            "action_type": action_type,
            "agent_type": "PPO",
            "h_network_type": h_network_rl_module.h_network_type.name,
            "reward_lambda": reward_lambda,
            "dataloader_type": sm_dl_train.__class__.__name__,
            "training_n_episodes": n_episodes,
            "trial_number": trial_number,
            "n_envs": n_envs,
            'agent_parameters': convert_PPO_params_to_dict(params),
            "seed": seed,
        }
        # additional details for discrete action type

        # logging experiment details
        save_experiment_details(trial_folder, experiment_details)
        env_train_vec.envs[0].env.save_environment_config(trial_folder)

        # train
        train_hnetwork_every_n_episodes = 5

        eval_callback = ValidateEveryNEpisodes(
            every_n_episodes=train_hnetwork_every_n_episodes * 5,  # every 5 updates to H-network
            sweep_every_n_episodes=-1,  # no sweep
            validation_log_folder=trial_folder / "logs_validation",
            validation_env=env_validation,
            h_network_rl_module=h_network_rl_module,
            best_model_save_path=trial_folder / "logs_validation" / "best_model",  # no best model saving in this case
            sample_n_episodes=20,     # randomly sample 20 episodes for evaluation
            enable_sweep=False,
            seed=seed,
            # optuna
            trial=trial,
        )

        env_logging_callback = EnvLoggingCallback(
            log_folder=trial_folder / "logs_train",
        )

        nan_encountered = False


        # reset for reproducibility
        h_network_rl_module.initialize_h_network_training()
        env_validation.reset(seed=seed); env_validation.action_space.seed(seed)

        try:
            rl_model.learn(
                total_timesteps=24 * 60 * n_episodes,  # n_episodes episodes, each episode is 24 hours (with 1 min sample frequency)
                progress_bar=False,
                tb_log_name="PPO_SmartMeterWorldContinuous",
                callback=[
                    UpdateGlobalTimestepCallback(),
                    env_logging_callback,
                    TrainHNetworkEveryNEpisodes(every_n_episodes=train_hnetwork_every_n_episodes, h_network_rl_module=h_network_rl_module),
                    eval_callback
                ]
            )
        except AssertionError as e:
            print_log(f"Encountered an error during training: {e}")
            nan_encountered = True
        finally:
            # free memory
            rl_model.env.close()
            env_validation.close()


        if nan_encountered:
            return float("nan")
        
        if eval_callback.optuna_is_pruned:
            raise optuna.exceptions.TrialPruned()
        
        # save the training results
        print_log("Saving fine-tuning results...")

        SmartMeterEnvironmentBase.save_episodes_rewards(env_logging_callback.get_episode_rewards(), trial_folder / "logs_train", TrainingMode.TRAIN)

        h_network_rl_module.save_train_loss_list(trial_folder / "logs_train" / "h_network_train_loss_list.json")

        # # save both RL model and H-network at the end of training
        rl_model_path = trial_folder / f"rl_model.zip"
        h_network_path = trial_folder / f"h_network2.pth" if h_network_rl_module.h_network_type == HNetworkType.H_NETWORK2 \
                        else trial_folder / f"h_network.pth"

        rl_model.save(rl_model_path)
        h_network_rl_module.save_h_network(h_network_path)

        return eval_callback.last_mean_reward


    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
        )
    except KeyboardInterrupt:
        pass

    print_log(f"Number of finished trials: {len(study.trials)}")

    print_log(f"Best trial:")
    best_trial = study.best_trial
    print_log(f"  Value: {best_trial.value}")
    print_log(f"  Params: ")
    for key, value in best_trial.params.items():
        print_log(f"    {key}: {value}")
    print_log(f"  User attrs: ")
    for key, value in best_trial.user_attrs.items():
        print_log(f"    {key}: {value}")

    

if __name__ == "__main__":

    torch.set_num_threads(1)  # set number of threads to  1 for faster training ??

    N_STARTUP_TRIALS = 5
    N_TRIALS = 50

    training_kwargs = {
        "reward_lambda": 0.5,
        "action_type": "continuous",
        "N_episodes": 400,  # total number of episodes to train
        "seed": 42,  # for reproducibility
    }

    rl_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    # rl_datetime = datetime(2025, 8, 5, 6, 31, 46).strftime("%Y%m%d_%H%M%S")  # for re-run existing trial
    experiment_folder = Path(REPO_DIR) / "experiments" / (rl_datetime + "_action_" + training_kwargs["action_type"] + "_reward_lambda_" + str(training_kwargs["reward_lambda"]) + "_optuna")
    if not experiment_folder.exists():
        experiment_folder.mkdir(parents=True)


    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps= 24*60*50)      # at least 50 episodes before pruning

    storage = optuna.storages.RDBStorage(url=f"sqlite:///{experiment_folder}/optuna_results.db")

    study = optuna.create_study(
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        study_name="PPO_finetuning_study",
        # load_if_exists=True,  # Load existing study if it exists
    )

    main(
        training_kwargs=training_kwargs,
        study=study
    )

