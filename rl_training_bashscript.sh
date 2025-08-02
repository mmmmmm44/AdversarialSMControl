#!/bin/bash

# reward_lambda_array=("0" "0.5" "1")
reward_lambda="0.5"
action_type_array=("continuous" "discrete")
# action_type_array=("discrete")
n_episodes=600

for action_type in "${action_type_array[@]}"; do
    echo "Running training for action type: $action_type"
    
    
    # Run the training script with the specified parameters
    python3 rl_training_script.py \
        --reward_lambda "$reward_lambda" \
        --action_type "$action_type" \
        --N_episodes "$n_episodes" \
        
done