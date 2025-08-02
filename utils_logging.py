import json

# --------------------
# logging related functions
# --------------------

def save_experiment_details(experiment_folder, experiment_details):
    # save the experiment details
    with open(experiment_folder / "experiment_details.json", 'w') as f:
        
        json.dump(experiment_details, f, indent=4)