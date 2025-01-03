import multiprocessing
import os
from pathlib import Path

import numpy as np

from experiments.running_time_prediction.config import CONFIG_TIMEOUT_CLASSIFICATION, \
    CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION, CONFIG_TIMEOUT_BASELINE
from src.running_time_prediction.timeout_classification import train_timeout_classifier_random_forest, \
    train_continuous_timeout_classifier, timeout_prediction_baseline
from src.util.constants import SUPPORTED_VERIFIERS, VERINET, OVAL, ABCROWN, ALL_EXPERIMENTS
from src.util.data_loaders import load_verinet_data, load_oval_bab_data, load_abcrown_data


def run_timeout_prediction_experiment(config: dict):
    """
    Run Timeout prediction experiments using a fixed feature collection phase from a config

    :param config: Refer to sample file experiments/running_time_prediction/config.py
    """

    verification_logs_path = Path(config.get("VERIFICATION_LOGS_PATH", "./verification_logs"))
    experiments = config.get("INCLUDED_EXPERIMENTS", None)
    if not experiments:
        experiments = ALL_EXPERIMENTS

    include_incomplete_results = config.get("INCLUDE_INCOMPLETE_RESULTS", True)

    results_path = config.get("RESULTS_PATH", "./results_running_time_prediction")
    os.makedirs(results_path, exist_ok=True)

    thresholds = config.get("TIMEOUT_CLASSIFICATION_THRESHOLDS", [0.5])

    random_state = config.get("RANDOM_STATE")

    for experiment in experiments:
        # skip hidden files
        if experiment.startswith("."):
            continue
        experiment_results_path = os.path.join(results_path, experiment)
        experiment_logs_path = os.path.join(verification_logs_path, experiment)
        experiment_info = config["EXPERIMENTS_INFO"].get(experiment)
        assert experiment_info, f"No Experiment Info for experiment {experiment} provided!"
        experiment_neuron_count = experiment_info.get("neuron_count")
        no_classes = experiment_info.get("no_classes", 10)
        assert experiment_neuron_count
        os.makedirs(experiment_results_path, exist_ok=True)

        feature_collection_cutoff = experiment_info.get("first_classification_at",
                                                        config.get("FEATURE_COLLECTION_CUTOFF", 10))

        for threshold in thresholds:
            for verifier in SUPPORTED_VERIFIERS:
                verifier_results_path = os.path.join(experiment_results_path, verifier)
                os.makedirs(verifier_results_path, exist_ok=True)
                if verifier == ABCROWN:
                    abcrown_log_file = os.path.join(experiment_logs_path, config.get("ABCROWN_LOG_NAME", "ABCROWN.log"))
                    if not os.path.isfile(abcrown_log_file):
                        print(f"Skipping verifier {verifier}! Log file {abcrown_log_file} not found!")
                        continue

                    features, running_times, results, enum_results = load_abcrown_data(
                        abcrown_log_file,
                        feature_collection_cutoff=feature_collection_cutoff,
                        neuron_count=experiment_neuron_count,
                        no_classes=no_classes,
                    )
                elif verifier == VERINET:
                    verinet_log_file = os.path.join(experiment_logs_path, config.get("VERINET_LOG_NAME", "VERINET.log"))
                    if not os.path.isfile(verinet_log_file):
                        print(f"Skipping verifier {verifier}! Log file {verinet_log_file} not found!")
                        continue
                    features, running_times, results, enum_results = load_verinet_data(
                        verinet_log_file,
                        neuron_count=experiment_neuron_count,
                        feature_collection_cutoff=feature_collection_cutoff,
                        filter_misclassified=True,
                        no_classes=no_classes
                    )
                elif verifier == OVAL:
                    oval_log_file = os.path.join(experiment_logs_path, config.get("OVAL_BAB_LOG_NAME", "OVAL-BAB.log"))
                    if not os.path.isfile(oval_log_file):
                        print(f"Skipping verifier {verifier}! Log file {oval_log_file} not found!")
                        continue
                    features, running_times, results, enum_results = load_oval_bab_data(oval_log_file,
                                                                                        neuron_count=experiment_neuron_count,
                                                                                        feature_collection_cutoff=feature_collection_cutoff,
                                                                                        filter_misclassified=True,
                                                                                        no_classes=no_classes)
                else:
                    # This should never happen!
                    assert 0, "Encountered Unknown Verifier!"

                train_timeout_classifier_random_forest(training_inputs=features, running_times=running_times,
                                                       verification_results=enum_results,
                                                       threshold=threshold,
                                                       include_incomplete_results=include_incomplete_results,
                                                       results_path=verifier_results_path,
                                                       feature_collection_cutoff=np.log10(feature_collection_cutoff),
                                                       random_state=random_state)


def run_continuous_timeout_prediction_experiment(config: dict):
    """
    Run Timeout prediction experiments using a continuous feature collection phase from a config

    :param config: Refer to sample file experiments/running_time_prediction/config.py
    """

    verification_logs_path = Path(config.get("VERIFICATION_LOGS_PATH", "./verification_logs"))
    experiments = config.get("INCLUDED_EXPERIMENTS", None)
    if not experiments:
        experiments = ALL_EXPERIMENTS

    include_incomplete_results = config.get("INCLUDE_INCOMPLETE_RESULTS", True)
    feature_collection_cutoff = config.get("FEATURE_COLLECTION_CUTOFF", 10)

    results_path = config.get("RESULTS_PATH", "./results_running_time_prediction")
    os.makedirs(results_path, exist_ok=True)

    thresholds = config.get("TIMEOUT_CLASSIFICATION_THRESHOLDS", [0.5])
    classification_frequency = config.get("TIMEOUT_CLASSIFICATION_FREQUENCY", 10)
    cutoff = config.get("MAX_RUNNING_TIME", 600)

    random_state = config.get("RANDOM_STATE", 42)

    num_workers = config.get("NUM_WORKERS", 4)

    args_queue = multiprocessing.Queue()

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
        first_classification_at = experiment_info.get("first_classification_at", config.get("FIRST_CLASSIFICATION_AT", classification_frequency))

        no_classes = experiment_info.get("no_classes", 10)
        os.makedirs(experiment_results_path, exist_ok=True)

        for threshold in thresholds:
            for verifier in SUPPORTED_VERIFIERS:
                print(f"---------------- VERIFIER {verifier} THRESHOLD {threshold} ---------------------------")
                verifier_results_path = os.path.join(experiment_results_path, verifier)
                os.makedirs(verifier_results_path, exist_ok=True)
                if verifier == ABCROWN:
                    log_path = os.path.join(experiment_logs_path, config.get("ABCROWN_LOG_NAME", "ABCROWN.log"))
                    load_data_func = load_abcrown_data
                elif verifier == VERINET:
                    log_path = os.path.join(experiment_logs_path, config.get("VERINET_LOG_NAME", "VERINET.log"))
                    load_data_func = load_verinet_data
                elif verifier == OVAL:
                    log_path = os.path.join(experiment_logs_path, config.get("OVAL_BAB_LOG_NAME", "OVAL-BAB.log"))
                    load_data_func = load_oval_bab_data
                else:
                    # This should never happen!
                    assert 0, "Encountered Unknown Verifier!"

                print(f"-------------------------- {experiment} -------------------")
                args = (
                    log_path, load_data_func, experiment_neuron_count, include_incomplete_results,
                    verifier_results_path,
                    threshold,
                    classification_frequency, cutoff, first_classification_at, no_classes, random_state)
                args_queue.put(args)

    procs = [multiprocessing.Process(target=train_continuous_timeout_classifier_worker, args=(args_queue,))
             for _ in range(num_workers)]
    for p in procs:
        p.daemon = True
        # poison pills
        args_queue.put(None)
        p.start()

    [p.join() for p in procs]


def train_continuous_timeout_classifier_worker(args_queue):
    while True:
        args = args_queue.get()
        if args is None:
            break
        train_continuous_timeout_classifier(*args)


def run_timeout_classification_experiments_from_config(config: dict):
    """
    Run Timeout prediction experiments using either fixed or continuous feature collection phase from a config. The
    differentiation is made based on if FEATURE_COLLECTION_CUTOFF in the config is an int (fixed cutoff) or
    equals "ADAPTIVE" (continuous cutoff).

    :param config: Refer to sample file experiments/running_time_prediction/config.py
    """

    if config.get("FEATURE_COLLECTION_CUTOFF") == "ADAPTIVE":
        run_continuous_timeout_prediction_experiment(config)
    else:
        run_timeout_prediction_experiment(config)


def run_baseline_heuristic_experiments_from_config(config: dict):
    """
    Runs a basic baseline for timeout prediction using a simple heuristic (if number of remaining branches multiplied
    by average time needed per branch exceeds cutoff time, predict timeout.
    :param config: Refer to sample file experiments/running_time_prediction/config.py
    """
    verification_logs_path = Path(config.get("VERIFICATION_LOGS_PATH", "./verification_logs"))
    experiments = config.get("INCLUDED_EXPERIMENTS", None)
    if not experiments:
        experiments = ALL_EXPERIMENTS

    include_incomplete_results = config.get("INCLUDE_INCOMPLETE_RESULTS", True)

    results_path = config.get("RESULTS_PATH", "./results_baseline_timeout_classification")
    os.makedirs(results_path, exist_ok=True)

    classification_frequency = config.get("TIMEOUT_CLASSIFICATION_FREQUENCY", 10)
    cutoff = config.get("MAX_RUNNING_TIME", 600)

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
        no_classes = experiment_info.get("no_classes", 10)
        os.makedirs(experiment_results_path, exist_ok=True)

        for verifier in SUPPORTED_VERIFIERS:
            verifier_results_path = os.path.join(experiment_results_path, verifier)
            os.makedirs(verifier_results_path, exist_ok=True)
            if verifier == ABCROWN:
                log_path = os.path.join(experiment_logs_path, config.get("ABCROWN_LOG_NAME", "ABCROWN.log"))
                if not os.path.isfile(log_path):
                    print(f"Skipping verifier {verifier}! Log file {log_path} not found!")
                    continue
                training_inputs, running_times, results, satisfiability_training_outputs = load_abcrown_data(
                    log_path, par=1,
                    neuron_count=experiment_neuron_count, feature_collection_cutoff=10, filter_misclassified=True,
                    frequency=classification_frequency)
                # filter relevant features
                for checkpoint, features in training_inputs.items():
                    training_inputs[checkpoint] = features[:, 7:9]
            elif verifier == VERINET:
                log_path = os.path.join(experiment_logs_path, config.get("VERINET_LOG_NAME", "VERINET.log"))
                if not os.path.isfile(log_path):
                    print(f"Skipping verifier {verifier}! Log file {log_path} not found!")
                    continue
                training_inputs, running_times, results, satisfiability_training_outputs = load_verinet_data(
                    log_path, par=1,
                    neuron_count=experiment_neuron_count, feature_collection_cutoff=10, filter_misclassified=True,
                    frequency=classification_frequency, no_classes=no_classes)
                # filter relevant features
                for checkpoint, features in training_inputs.items():
                    training_inputs[checkpoint] = features[:, 12:14]
            elif verifier == OVAL:
                log_path = os.path.join(experiment_logs_path, config.get("OVAL_BAB_LOG_NAME", "OVAL-BAB.log"))
                if not os.path.isfile(log_path):
                    print(f"Skipping verifier {verifier}! Log file {log_path} not found!")
                    continue
                training_inputs, running_times, results, satisfiability_training_outputs = load_oval_bab_data(
                    log_path, par=1,
                    neuron_count=experiment_neuron_count, feature_collection_cutoff=10, filter_misclassified=True,
                    frequency=classification_frequency)

                for checkpoint, features in training_inputs.items():
                    training_inputs[checkpoint] = features[:, 11:14]
            else:
                # This should never happen!
                assert 0, "Encountered Unknown Verifier!"

            timeout_prediction_baseline(features=training_inputs, running_times=running_times,
                                        verification_results=satisfiability_training_outputs, verifier=verifier,
                                        include_incomplete_results=include_incomplete_results,
                                        results_path=verifier_results_path,
                                        classification_frequency=classification_frequency, cutoff=cutoff)


if __name__ == "__main__":
    run_timeout_classification_experiments_from_config(CONFIG_TIMEOUT_CLASSIFICATION)
    run_timeout_classification_experiments_from_config(CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION)
    # run_timeout_classification_experiments_from_config(CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION)
    # run_baseline_heuristic_experiments_from_config(CONFIG_TIMEOUT_BASELINE)
