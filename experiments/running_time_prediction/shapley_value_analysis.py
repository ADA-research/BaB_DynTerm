import multiprocessing
import os
from pathlib import Path

from experiments.running_time_prediction.config import CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION
from src.feature_ablation_study.custom_shapley_plots import aggregate_over_all_benchmarks
from src.feature_ablation_study.shapley_experiments import train_continuous_timeout_classifier_shapley_worker
from src.util.constants import SUPPORTED_VERIFIERS, ABCROWN, VERINET, OVAL
from src.util.data_loaders import load_abcrown_data, load_verinet_data, load_oval_bab_data


def run_shapley_value_study_dynamic_timeout_termination(config, thresholds=None, results_path=None):
    """
        Run Timeout prediction experiments using a continuous feature collection phase from a config

        :param config: Refer to sample file experiments/running_time_prediction/config.py
        """

    verification_logs_path = Path(config.get("VERIFICATION_LOGS_PATH", "./verification_logs"))
    experiments = config.get("INCLUDED_EXPERIMENTS", None)
    if not experiments:
        experiments = os.listdir(verification_logs_path)

    include_incomplete_results = config.get("INCLUDE_INCOMPLETE_RESULTS", True)
    feature_collection_cutoff = config.get("FEATURE_COLLECTION_CUTOFF", 10)

    results_path = './results/feature_ablation/shapley_continuous_classification/' if not results_path else results_path
    os.makedirs(results_path, exist_ok=True)

    thresholds = config.get("TIMEOUT_CLASSIFICATION_THRESHOLDS", [0.5]) if not thresholds else thresholds
    classification_frequency = config.get("TIMEOUT_CLASSIFICATION_FREQUENCY", 10)
    cutoff = config.get("MAX_RUNNING_TIME", 600)

    random_state = config.get("RANDOM_STATE", 42)

    num_workers = config.get("NUM_WORKERS", multiprocessing.cpu_count())

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
        first_classification_at = experiment_info.get("first_classification_at", classification_frequency)

        no_classes = experiment_info.get("no_classes", 10)
        os.makedirs(experiment_results_path, exist_ok=True)

        for threshold in thresholds:
            for verifier in SUPPORTED_VERIFIERS:
                print(f"---------------- VERIFIER {verifier} THRESHOLD {threshold} ---------------------------")
                verifier_results_path = os.path.join(experiment_results_path, verifier)
                os.makedirs(verifier_results_path, exist_ok=True)
                if verifier == ABCROWN:
                    log_path = os.path.join(experiment_logs_path, config.get("ABCROWN_LOG_NAME", "ABCROWN.log"))
                    if not os.path.isfile(log_path):
                        print(f"Skipping verifier {verifier}! Log file {log_path} not found!")
                        continue
                    load_data_func = load_abcrown_data
                elif verifier == VERINET:
                    log_path = os.path.join(experiment_logs_path, config.get("VERINET_LOG_NAME", "VERINET.log"))
                    if not os.path.isfile(log_path):
                        print(f"Skipping verifier {verifier}! Log file {log_path} not found!")
                        continue
                    load_data_func = load_verinet_data
                elif verifier == OVAL:
                    log_path = os.path.join(experiment_logs_path, config.get("OVAL_BAB_LOG_NAME", "OVAL-BAB.log"))
                    if not os.path.isfile(log_path):
                        print(f"Skipping verifier {verifier}! Log file {log_path} not found!")
                        continue
                    load_data_func = load_oval_bab_data
                else:
                    # This should never happen!
                    assert 0, "Encountered Unknown Verifier!"

                print(f"-------------------------- {experiment} -------------------")
                args = (
                    log_path, load_data_func, experiment_neuron_count, include_incomplete_results, verifier_results_path,
                    threshold,
                    classification_frequency, cutoff, first_classification_at, no_classes, random_state, verifier)

                args_queue.put(args)
                # train_continuous_timeout_classifier_feature_ablation(log_path=log_path, load_data_func=load_data_func,
                #                                     neuron_count=experiment_neuron_count,
                #                                     include_incomplete_results=include_incomplete_results,
                #                                     results_path=verifier_results_path, threshold=threshold,
                #                                     classification_frequency=classification_frequency, cutoff=cutoff,
                #                                     first_classification_at=first_classification_at,
                #                                     no_classes=no_classes,
                #                                     random_state=random_state, verifier=verifier)

    procs = [multiprocessing.Process(target=train_continuous_timeout_classifier_shapley_worker, args=(args_queue, ))
             for _ in range(num_workers)]
    for p in procs:
        p.daemon = True
        args_queue.put(None)
        p.start()

    [p.join() for p in procs]

    aggregate_over_all_benchmarks(results_path=results_path)





if __name__ == "__main__":
    run_shapley_value_study_dynamic_timeout_termination(CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION, thresholds=[.99],
                                                        results_path='./results/shapley_value_study')
