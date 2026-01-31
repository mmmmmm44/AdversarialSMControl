# Battery-Aided Load Shaping Strategies based on Reinforcement Learning for Optimal Privacy-Cost Trade-off in Smart Electricity Metering

## Abstract

The rising popularity of smart meters arised privacy concern on its fine-grained electricity consumption data collection. Sharing such data with utility providers may expose household members’ private information on its usage habits, potentially leading to unwanted surveillance and profiling.
One promising approach to reduce private information leakage from the smart meter data is to use load shaping strategies. For example, one may create artificial grid load by using an energy management unit (EMU) with a rechargeable battery (RB) to mask the household’s load. Previous studies have shown that the EMU policy can be learnt using reinforcement learning (RL) with a mutual information (MI)-based reward signal. However, its adaptation is limited on quantized household load and charging/discharging power, and low sample rate.
To address this limitation, we extend the EMU policy with MI-based reward signal to support continuous household load and charging/discharging power on a relatively high sample rate. The approach is implemented with a policy gradient algorithm namely proximal policy optimization (PPO). Performance of the new algorithm (PPO-MI) is evaluated using an actual SMs dataset and compared with its state-of-the-art quantized counterpart (DDQL-MI). Our results show significant improvement over its quantized counterpart in both privacy and cost metrics. PPO-MI achieved 69.24% reduction in average MI compared to DDQL-MI under balanced privacy- cost trade-off, while reducing the incurred extra electricity cost by 18.36%. This work will be submitted to the IEEE Transactions on Smart Grid, with the draft paper available in Appendix G.

## How to Start

## Environment Setup

Setup the conda environment using `environment.yml`.

## How to use

1. We have to create the dataset for this project. For info about dataset creation, please go to [README](dataset/README.md)

2. After creating the dataset, to train both PPO-MI and DDQL-MI on our experiment settings, execute

```
rl_training_bashscript.sh
```

It will train the both RL algorithms with `lambda` = 0, 0.5 and 1. Replace the content in `reward_lambda_array` and `action_type_array` to train with your own lambda value and network ("discrete" = DDQL-MI, "continuous" = PPO-MI)

3. One may want to perform its own hyperparameter search for PPO-MI. Run the below command to perform hyperparameter search on PPO-MI with `lambda` = 0.5.

```
python rl_training_script_PPO_finetuning.py
```

Then replace line 124 to 133 with your own hyperparameters.

4. During training, one may want to visualize the progress. Since the environment has tensorboard installed, we can create a new shell session, then execute the following command.

```
cd experiments
tensorboard .
```

After training all six models, we can also use tensorboard to have a brief idea of the models' performance.

```
cd experiments
tensorboard --logdir_spec=PPO-MI-0.0:<yr_datetime>_action_continuous_reward_lambda_0.0/,PPO-MI-0.5:<yr_datetime>_action_continuous_reward_lambda_0.5/,PPO-MI-1.0:<yr_datetime>_action_continuous_reward_lambda_1.0/,DDQL-MI-0.0:<yr_datetime>_action_discrete_reward_lambda_0.0/,DDQL-MI-0.5:<yr_datetime>_action_discrete_reward_lambda_0.5/,DDQL-MI-1.0:<yr_datetime>_action_discrete_reward_lambda_1.0/
```

5. To produce numerious graphs for detailed analysis, please make use of the notebooks that begin with `expt_results_` and `results_<multi|per>`. Below I will briefly describe the purpose of each notebook

    |file name|purpose|
    |---|---|
    |expt_results_cross_model_hnetwork.ipynb|Create a single graph to plot the mean & sd of loss of the H-network of all models during training|
    |expt_results_cross_models_multi_episodes.ipynb|Create a grid plot to describe how a RL model evolves during training|
    |expt_results_cross_models_privacy_protection_logtest.ipynb|Create two graphs that describes the distribution of estimated MI and extra cost incurred of all models, under test set|
    |expt_results_cross_models_privacy_protection.ipynb|Create two graphs that describes the distribution of estimated MI and extra cost incurred of all models, under training set|
    |expt_results_cross_models.ipynb|Create graphs that report the reward, f-signal, g-signal of all models, and make comparisons between models under same lambda value.
    |results_multi_episodes_plot.ipynb|Create plots of a single episode under different training iteration during training. Great to visualize performance changes during training under the same episode|
    |results_per_episode_plot.ipynb|It plots various graphs to each training episode, from user load vs grid load vs battery state-of-charge, to the estimated MI and PSD.|
    |results_multi_episodes_plot_20250922.ipynb| (Same as above). Note that it is targeted for a selected checkpoint during training.|

    When reading the _results_multi_episodes_plot_20250922.ipynb_, follow the _How to use_ section to execute either _rl_training_eval_DDQN.ipynb_ or _rl_training_eval_PPO.ipynb_ depending your model type.
We first create the dataset, then execute the training scripts.

### Creating the dataset

1. Download the UK-DALE dataset (Disaggregated (6s) appliance power and aggregated (1s) whole house power) from [UK-DALE dataset download page](https://dap.ceda.ac.uk/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/ReadMe_DALE-2017.html). Put the downloaded `UK-DALE-2017.tgz` under `./datasets` by creating a `datasets` folder.

2. Unzip the `UK-DALE-2017.tgz` folder, and undzip the `ukdale.zip` file. The resulting folder strucutre should be like this

    ```text
    datasets
    |-- UK-DALE-FULL-disaggregated
    |   |-- ukdale
    |   |   |-- house_1
    ```

3. Create a conda virtual environment using `environment.yml`. Activate the environment.

4. Run `01_data_cleaning.ipynb`

5. Run `02_build_load_signature.ipynb`

6. Run `03_data_split.ipynb`. For more information, please go to [dataset/README](dataset/README.md)

7. Run `04_downsampling.ipynb`

### Performing experiments

1. To train the PPO-MI models, open `rl_training_bashscript.sh`. Comment line 5 and uncomment line 6. I.e.

    `rl_training_bashscript.sh`

    ```bash
    #!/bin/bash

    reward_lambda_array=("0" "0.5" "1")
    # reward_lambda="0.5"
    # action_type_array=("discrete")
    action_type_array=("continuous")
    n_episodes=800
    seed=42
    ```

    Then activate the virtual environment, and execute this script under the project root directory.

2. To train the DDQL-MI models, open `rl_training_bashscript.sh`. Comment line 6 and uncomment line 5. I.e.

    `rl_training_bashscript.sh`

    ```bash
    #!/bin/bash

    reward_lambda_array=("0" "0.5" "1")
    # reward_lambda="0.5"
    action_type_array=("discrete")
    # action_type_array=("continuous")
    n_episodes=800
    seed=42
    ```

    Then activate the virtual environment, and execute this script under the project root directory.

3. To execute hyperparameter optimization script for PPO-MI, execute `rl_training_script_PPO_finetuning.py` for 30 trials. This should take around 2 days on a single RTX4090.

4. Training results can be analyzed using

    - `expt_results_cross_models.ipynb`
    - `expt_results_cross_models_privacy_protection.ipynb`
    - `expt_results_cross_models_privacy_protection_logtest.ipynb`
    - `expt_results_cross_models_multi_episodes.ipynb`
    - `expt_results_cross_models_hnetwork.ipynb`
