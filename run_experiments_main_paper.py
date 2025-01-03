from experiments.running_time_prediction.config import CONFIG_DYNAMIC_ALGORITHM_TERMINATION
from experiments.running_time_prediction.run_timeout_classification_experiments import \
    run_timeout_classification_experiments_from_config

if __name__ == '__main__':
    run_timeout_classification_experiments_from_config(CONFIG_DYNAMIC_ALGORITHM_TERMINATION)

