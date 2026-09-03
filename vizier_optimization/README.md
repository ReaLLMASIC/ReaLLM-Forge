# OpenVizier use
# Dependence: google-vizier[all] and torch
# Install(take ~20mins): 
# pip install google-vizier[all]
# pip install torch

# Parmeters:
# vizier_iterations: determine how many trails to run
# vizier_algorithm: choose which algorithm to use
# objectives: choose which objectives to use, possible choices: ["validation_loss", "num_bits"]
# c: config file, in yaml format