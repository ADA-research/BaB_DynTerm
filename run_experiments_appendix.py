from experiments.running_time_prediction.config import CONFIG_RUNNING_TIME_REGRESSION, \
    CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION_APPENDIX, CONFIG_DYNAMIC_ALGORITHM_TERMINATION
from experiments.running_time_prediction.run_running_time_regression_experiments import run_experiments_from_config
from experiments.running_time_prediction.run_timeout_classification_experiments import \
    run_timeout_classification_experiments_from_config
from experiments.running_time_prediction.shapley_value_analysis import \
    run_shapley_value_study_dynamic_timeout_termination

if __name__ == '__main__':
    run_experiments_from_config(CONFIG_RUNNING_TIME_REGRESSION)
    run_timeout_classification_experiments_from_config(CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION_APPENDIX)
    run_shapley_value_study_dynamic_timeout_termination(CONFIG_DYNAMIC_ALGORITHM_TERMINATION, thresholds=[.99],
                                                        results_path='./results/shapley_value_study')


