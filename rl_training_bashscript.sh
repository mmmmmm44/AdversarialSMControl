#!/bin/bash

reward_lambda_array=("0" "0.5" "1")
# reward_lambda="0.5"
action_type_array=("discrete")
# action_type_array=("continuous")
n_episodes=800
seed=42

for action_type in "${action_type_array[@]}"; do
    echo "Running training for action type: $action_type"
    
    for reward_lambda in "${reward_lambda_array[@]}"; do
        echo "Using reward lambda: $reward_lambda"
    
        # Run a specific training script for each type of action
        if [ "$action_type" == "discrete" ]; then
            echo "Using discrete action type"

            python3 rl_training_script_DDQN.py \
                --reward_lambda "$reward_lambda" \
                --action_type "$action_type" \
                --N_episodes "$n_episodes" \
                --seed "$seed"

        elif [ "$action_type" == "continuous" ]; then
            echo "Using continuous action type"

            python3 rl_training_script_PPO.py \
                --reward_lambda "$reward_lambda" \
                --action_type "$action_type" \
                --N_episodes "$n_episodes" \
                --seed "$seed"
        else
            echo "Unknown action type: $action_type"
            exit 1
        fi

        echo "Training completed for action type: $action_type with reward lambda: $reward_lambda for $n_episodes episodes"

    done
        
done