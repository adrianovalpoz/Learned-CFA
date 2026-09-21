"""Single place where every folder of the repository is defined.

Only folders are listed here: file paths are built where they are used, for example
os.path.join(save_NN_dir, 'static_cfa.json').

All folders are derived from the position of this file, so the repository works from
any location it is cloned into, on any operating system. No absolute path is ever
written by hand: to move a folder, edit the line here and nothing else.

Output folders are not tracked by git, so they may be missing on a fresh clone:
the code that writes into them must create them first with os.makedirs(..., exist_ok=True).

Usage:
    from learned_cfa_src.directory_names import data_json_dir, save_NN_dir
"""

import os

# Directory path creation
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root

# %% ############################## SOURCE CODE ##############################
package_dir = os.path.join(base_dir, 'learned_cfa_src')                  # method source code
milp_dir = os.path.join(package_dir, 'milp')                             # MILP formulation and system models
evolution_strategy_dir = os.path.join(package_dir, 'evolution_strategy') # policy and ES training
data_pipeline_dir = os.path.join(package_dir, 'data_pipeline')           # download, cleaning, scenario generation
postprocessing_dir = os.path.join(base_dir, 'postprocessing')            # scripts that build the paper artefacts
docs_dir = os.path.join(base_dir, 'docs')                                # reproducibility notes

# %% ############################## CONFIGURATION ##############################
config_dir = os.path.join(base_dir, 'config')                            # hand-editable configuration files

# %% ############################## DATA ##############################
data_dir = os.path.join(base_dir, 'data')                                # dataset root
prices_dir = os.path.join(data_dir, 'prices')                            # day-ahead prices (raw input, not shipped)
imbalance_dir = os.path.join(data_dir, 'imbalance')                      # imbalance prices (raw input, not shipped)
json_repository_dir = os.path.join(data_dir, 'json_repository_for_simulation')
data_json_dir = os.path.join(json_repository_dir, 'SystemData')          # one JSON per day: system, forecasts, prices
scenario_json_dir = os.path.join(json_repository_dir, 'ScenarioData')    # one JSON per day: scenario MILP results
sample_data_dir = os.path.join(data_dir, 'sample')                       # few days shipped with the code

# %% ############################## TRAINED MODELS ##############################
save_NN_dir = os.path.join(base_dir, 'saved_ANN')                        # directory where the trained policies are stored
NN_size_dir = os.path.join(evolution_strategy_dir, 'NN_Size')            # hyperparameter search output

# %% ############################## OUTPUTS ##############################
results_dir = os.path.join(base_dir, 'results')                          # everything the code regenerates
training_check_dir = os.path.join(results_dir, 'TrainingCheck')          # per-iteration figures and training log
fig_dir = os.path.join(results_dir, 'figure')                            # figures produced by postprocessing
