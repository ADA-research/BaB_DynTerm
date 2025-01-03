import os
from pathlib import Path

import numpy as np

from src.running_time_prediction.running_time_regression import train_running_time_predictor_random_forest
from experiments.running_time_prediction.config import CONFIG_RUNNING_TIME_REGRESSION
from src.util.constants import SUPPORTED_VERIFIERS, ABCROWN, VERINET, OVAL, ALL_EXPERIMENTS
from src.util.data_loaders import load_abcrown_data, load_verinet_data, load_oval_bab_data


def run_running_time_regression_experiment(features, running_times, results, include_timeouts=True,
                                           include_incomplete_results=True, results_path="./results",
                                           feature_collection_cutoff=None, random_state=42):
    """
    Starts running time regression experiment

    :param features: numpy array of features for each problem instance
    :param running_times: numpy array of running times for each problem instance
    :param results: numpy array of verification results
    :param include_timeouts: If to include/exclude timeouted instances from training/prediction
    :param include_incomplete_results: If to include/exclude instances solved very quickly using incomplete methods from training/prediction
    :param results_path: Path to save results to
    :param feature_collection_cutoff: Number of seconds to collect features for before performing running time prediction
    :param random_state: random state to for random forest/five-fold-cross-validation splits
    """
    train_running_time_predictor_random_forest(training_inputs=features, running_times=running_times,
                                               verification_results=results, include_timeouts=include_timeouts,
                                               include_incomplete_results=include_incomplete_results,
                                               results_path=results_path,
                                               feature_collection_cutoff=feature_collection_cutoff,
                                               random_state=42)


def run_experiments_from_config(config):
    """
    Function to run running time regression experiments with a provided config
    :param config: Refer to sample file experiments/running_time_prediction/config.py
    :type config: dict
    :return: None
    """

    verification_logs_path = Path(config.get("VERIFICATION_LOGS_PATH", "./verification_logs"))
    experiments = config.get("INCLUDED_EXPERIMENTS", None)
    if not experiments:
        experiments = ALL_EXPERIMENTS

    include_timeouts = config.get("INCLUDE_TIMEOUTS", True)
    include_incomplete_results = config.get("INCLUDE_INCOMPLETE_RESULTS", True)
    cutoff = config.get("MAX_RUNNING_TIME", None)

    results_path = config.get("RESULTS_PATH", "./results_running_time_regression")
    os.makedirs(results_path, exist_ok=True)

    for experiment in experiments:
        # skip hidden files
        if experiment.startswith("."):
            continue
        experiment_results_path = os.path.join(results_path, experiment)
        experiment_logs_path = os.path.join(verification_logs_path, experiment)
        experiment_info = config["EXPERIMENTS_INFO"].get(experiment)
        assert experiment_info, f"No Experiment Info for experiment {experiment} provided!"
        experiment_neuron_count = experiment_info.get("neuron_count")
        assert experiment_neuron_count
        os.makedirs(experiment_results_path, exist_ok=True)
        feature_collection_cutoff = experiment_info.get("first_classification_at",
                                                        config.get("FEATURE_COLLECTION_CUTOFF", 10))
        random_state = experiment_info.get("RANDOM_STATE", 42)

        print(f"--------------- Running Running Time Prediction Experiment {experiment} ------------------------------")

        for verifier in SUPPORTED_VERIFIERS:
            verifier_results_path = os.path.join(experiment_results_path, verifier)
            os.makedirs(verifier_results_path, exist_ok=True)
            if verifier == ABCROWN:
                abcrown_log_file = os.path.join(experiment_logs_path,
                                                config.get("ABCROWN_LOG_NAME", "ABCROWN.log"))

                features, running_times, results, enum_results = load_abcrown_data(
                    abcrown_log_file,
                    feature_collection_cutoff=feature_collection_cutoff,
                    neuron_count=experiment_neuron_count,
                    artificial_cutoff=cutoff
                )
            elif verifier == VERINET:
                verinet_log_file = os.path.join(experiment_logs_path,
                                                config.get("VERINET_LOG_NAME", "VERINET.log"))
                features, running_times, results, enum_results = load_verinet_data(
                    verinet_log_file,
                    neuron_count=experiment_neuron_count,
                    feature_collection_cutoff=feature_collection_cutoff,
                    filter_misclassified=True,
                    artificial_cutoff=cutoff
                )
            elif verifier == OVAL:
                oval_log_file = os.path.join(experiment_logs_path,
                                             config.get("OVAL_BAB_LOG_NAME", "OVAL-BAB.log"))

                features, running_times, results, enum_results = load_oval_bab_data(
                    oval_log_file,
                    neuron_count=experiment_neuron_count,
                    feature_collection_cutoff=feature_collection_cutoff,
                    filter_misclassified=True,
                    artificial_cutoff=cutoff
                )
            else:
                # This should never happen!
                assert 0, "Encountered Unknown Verifier!"

            run_running_time_regression_experiment(features=features, running_times=running_times, results=enum_results,
                                                   include_timeouts=include_timeouts,
                                                   include_incomplete_results=include_incomplete_results,
                                                   results_path=verifier_results_path,
                                                   feature_collection_cutoff=np.log10(feature_collection_cutoff),
                                                   random_state=random_state)


if __name__ == "__main__":
    run_experiments_from_config(CONFIG_RUNNING_TIME_REGRESSION)
