# UoM Dissertation project folder

## Description

The project aims to improve the algorithm, DDQL-MI, presented in the paper "Privacy-Cost Management in Smart Meters With Mutual-Information-Based Reinforcement Learning" by introducing new assumptions to the derivation of the per-step privacy signal. This results in a continuous probability distribution learnt by the H-network in the paper, allows a RL algorithm to produce charging/discharging rate in a continuous action space, instead of a discrete action space. A custom implementation of the DDQL-MI following available information in this paper and another paper from the same author is provided as a baseline for comparison.

A custom RL environment mimicking the energy management unit is produced, which includes a rechargeable battery with adjustable efficiency, a H-network to be trained along with.

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