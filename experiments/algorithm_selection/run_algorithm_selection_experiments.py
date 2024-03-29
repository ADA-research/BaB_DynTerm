import os
from pathlib import Path

from experiments.algorithm_selection.config import CONFIG_ADAPTIVE_ALGORITHM_SELECTION
from src.algorithm_selection.algorithm_selection import adaptive_algorithm_selection
from src.util.constants import SUPPORTED_VERIFIERS, ABCROWN, VERINET, OVAL
from src.util.data_loaders import load_algorithm_selection_data


def run_algorithm_selection_experiment_from_config(config):
    verification_logs_path = Path(config.get("VERIFICATION_LOGS_PATH", "./verification_logs"))
    experiments = config.get("INCLUDED_EXPERIMENTS", os.listdir(verification_logs_path))
    if not experiments:
        experiments = os.listdir(verification_logs_path)

    feature_collection_cutoff = config.get("FEATURE_COLLECTION_CUTOFF")
    frequency = config.get("ALGORITHM_SELECTION_FREQUENCY")
    cutoff = config.get("MAX_RUNNING_TIME")
    par = config.get("PAR", 1)
    stop_predicted_timeouts = config.get("STOP_PREDICTED_TIMEOUTS", False)
    selection_thresholds = config.get("SELECTION_THRESHOLDS", [.99])

    classification_method = config.get("CLASSIFICATION_METHOD", "NAIVE")

    results_path = config.get("RESULTS_PATH", "./results_running_time_prediction")
    os.makedirs(results_path, exist_ok=True)

    for experiment in experiments:
        verifiers = SUPPORTED_VERIFIERS.copy()

        experiment_results_path = os.path.join(results_path, experiment)
        experiment_logs_path = os.path.join(verification_logs_path, experiment)
        experiment_info = config["EXPERIMENTS_INFO"].get(experiment)
        assert experiment_info, f"No Experiment Info for experiment {experiment} provided!"
        experiment_neuron_count = experiment_info.get("neuron_count")
        assert experiment_neuron_count
        first_classification_at = experiment_info.get("first_classification_at", frequency)
        no_classes = experiment_info.get("no_classes", 10)
        os.makedirs(experiment_results_path, exist_ok=True)

        abcrown_log_file = os.path.join(experiment_logs_path, config.get("ABCROWN_LOG_NAME", "ABCROWN.log"))
        if not os.path.isfile(abcrown_log_file):
            print(f"Not including verifier {ABCROWN}! Log file {abcrown_log_file} not found!")
            abcrown_log_file = None
            verifiers.remove(ABCROWN)

        verinet_log_file = os.path.join(experiment_logs_path, config.get("VERINET_LOG_NAME", "VERINET.log"))
        if not os.path.isfile(verinet_log_file):
            print(f"Not including verifier {VERINET}! Log file {verinet_log_file} not found!")
            verinet_log_file = None
            verifiers.remove(VERINET)

        oval_log_file = os.path.join(experiment_logs_path, config.get("OVAL_BAB_LOG_NAME", "OVAL-BAB.log"))
        if not os.path.isfile(oval_log_file):
            print(f"Not including verifier {OVAL}! Log file {oval_log_file} not found!")
            oval_log_file = None
            verifiers.remove(OVAL)

        features, running_times, results, enum_results, features_best_verifiers, best_verifiers, verifier_data = load_algorithm_selection_data(
            abcrown_log_file=abcrown_log_file, verinet_log_file=verinet_log_file, oval_log_file=oval_log_file,
            feature_collection_cutoff=feature_collection_cutoff, frequency=frequency,
            neuron_count=experiment_neuron_count, cutoff=cutoff, par=par, no_classes=no_classes)

        for threshold in selection_thresholds:
            adaptive_algorithm_selection(
                features=features_best_verifiers,
                best_verifiers=best_verifiers,
                enum_results=enum_results,
                running_times=running_times,
                verifier_data=verifier_data,
                frequency=frequency,
                threshold=threshold,
                artificial_cutoff=cutoff,
                verifiers=verifiers,
                feature_collection_cutoff=feature_collection_cutoff,
                stop_predicted_timeouts=stop_predicted_timeouts,
                results_path=experiment_results_path,
                first_classification_at=first_classification_at,
                classification_method=classification_method
            )


if __name__ == "__main__":
    run_algorithm_selection_experiment_from_config(CONFIG_ADAPTIVE_ALGORITHM_SELECTION)
