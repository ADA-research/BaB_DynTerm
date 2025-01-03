import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import multiprocessing

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from experiments.running_time_prediction.config import CONFIG_TIMEOUT_CLASSIFICATION, \
    CONFIG_DYNAMIC_ALGORITHM_TERMINATION
from src.eval.running_time_prediction import eval_final_timeout_classification, eval_timeout_classification_fold

from src.util.constants import SUPPORTED_VERIFIERS, ABCROWN, VERINET, OVAL, VERIFIER_FEATURE_MAP, OVAL_FEATURE_NAMES, \
    VERINET_FEATURE_NAMES, ABCROWN_FEATURE_NAMES, TIMEOUT, experiment_samples, experiment_groups, UNSAT, ALL_EXPERIMENTS
from src.util.data_loaders import load_verinet_data, load_oval_bab_data, load_abcrown_data


def train_continuous_timeout_classifier_feature_ablation(log_path, load_data_func, neuron_count=None,
                                                         include_incomplete_results=False,
                                                         results_path="./results", threshold=.5,
                                                         classification_frequency=10, cutoff=600,
                                                         first_classification_at=None,
                                                         no_classes=10, random_state=42, verifier=ABCROWN):
    print("---------------------- TRAINING RANDOM FOREST TIMEOUT CLASSIFIER ------------------------")

    training_inputs, running_time_training_outputs, results, satisfiability_training_outputs = load_data_func(
        log_path, par=1,
        neuron_count=neuron_count, feature_collection_cutoff=10, filter_misclassified=True,
        frequency=classification_frequency, no_classes=no_classes)

    if training_inputs is None or running_time_training_outputs is None or results is None or satisfiability_training_outputs is None:
        print(f"Skipping Experiment for {log_path} - Features or Logs could not be found! ")
        return

    timeout_indices = np.where(satisfiability_training_outputs == 2)
    no_timeout_indices = np.where(satisfiability_training_outputs != 2)
    sat_timeout_labels = np.copy(satisfiability_training_outputs)
    sat_timeout_labels[timeout_indices] = 1
    sat_timeout_labels[no_timeout_indices] = 0

    if not include_incomplete_results:
        print("--------------------------- EXCLUDING INCOMPLETE RESULTS FROM TRAINING ------------------------------")
        no_incomplete_indices = np.where(running_time_training_outputs > np.log10(10))
        training_inputs = training_inputs[no_incomplete_indices]
        running_time_training_outputs = running_time_training_outputs[no_incomplete_indices]
        satisfiability_training_outputs = satisfiability_training_outputs[no_incomplete_indices]
        sat_timeout_labels = sat_timeout_labels[no_incomplete_indices]

    # Fixed Random State for Comparability between fixed and dynamic Timeout Classification
    kf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)

    feature_collection_cutoff = np.log10(first_classification_at) if first_classification_at else np.log10(
        classification_frequency)

    split_labels = np.array([1 for _ in satisfiability_training_outputs])
    split_labels[running_time_training_outputs < feature_collection_cutoff] = 0
    split_labels[satisfiability_training_outputs == 2.] = 2

    for feature_index, feature in enumerate(VERIFIER_FEATURE_MAP[verifier]):
        results_path_feature = f"{results_path}/{feature}/"
        os.makedirs(results_path_feature, exist_ok=True)

        preds = np.array([])
        sat_labels_shuffled = np.array([])
        running_time_labels_shuffled = np.array([])
        sat_timeout_labels_shuffled = np.array([])
        metrics = {}
        running_time_labels_timeout_prediction = np.array([])

        feature_selector = [x for x in range(training_inputs[classification_frequency].shape[1]) if x != feature_index]

        print(f"EXCLUDING FEATURE {feature} ON VERIFIER {verifier}")

        for fold, (train_index, test_index) in enumerate(
                kf.split(training_inputs[classification_frequency], split_labels)):
            train_labels_sat = sat_timeout_labels[train_index]
            test_labels_sat = sat_timeout_labels[test_index]
            running_times_timeout_prediction_test = running_time_training_outputs[test_index].copy()
            test_running_times = running_time_training_outputs[test_index]

            print(f"---------------------------------- Fold {fold} ----------------------------------")

            solved_instances = np.array([], dtype=int)
            stopped_instances = np.array([], dtype=int)

            for checkpoint in range(classification_frequency, cutoff, classification_frequency):
                training_inputs_checkpoint = training_inputs[checkpoint]

                upper_bound_running_time = np.log10(checkpoint)
                lower_bound_running_time = np.log10(checkpoint - classification_frequency)
                solved_instances_checkpoint = np.where(
                    (test_running_times >= lower_bound_running_time) & (
                            test_running_times <= upper_bound_running_time) & (
                            satisfiability_training_outputs[test_index] != 2))[0]
                if solved_instances_checkpoint.shape[0] > 0:
                    print(f"SOLVED {solved_instances_checkpoint} at {checkpoint} s!")
                solved_instances = np.append(solved_instances, solved_instances_checkpoint)

                if len(test_index) - len(stopped_instances) - len(solved_instances) == 0:
                    print("All Test Samples stopped or solved!")
                    break

                if first_classification_at and checkpoint < first_classification_at:
                    print(f"Skipping Timeout Prediction at checkpoint {checkpoint} as specified in config!")
                    continue

                train_inputs = training_inputs_checkpoint[train_index][:, feature_selector]
                test_inputs = training_inputs_checkpoint[test_index][:, feature_selector]
                rf_classifier = RandomForestClassifier(n_estimators=200, random_state=random_state)
                rf_classifier.fit(train_inputs, train_labels_sat)

                probability_predictions = rf_classifier.predict_proba(test_inputs)
                if probability_predictions.shape[1] > 1:
                    y_pred_custom_threshold = (probability_predictions[:, 1] > threshold).astype(int)
                    predictions = y_pred_custom_threshold
                else:
                    predictions = probability_predictions
                stopped_instances_checkpoint = []
                for index, test_sample in enumerate(test_index):
                    if predictions[index] == 1 and not np.isin(index, stopped_instances) and not np.isin(index,
                                                                                                         solved_instances):
                        stopped_instances_checkpoint.append(index)
                print(f"STOPPED INSTANCES {stopped_instances_checkpoint} at {checkpoint}s!")
                stopped_instances = np.append(stopped_instances, np.array(stopped_instances_checkpoint, dtype=int))
                running_times_timeout_prediction_test[stopped_instances_checkpoint] = np.log10(checkpoint)

            predictions = np.zeros(test_labels_sat.shape)
            predictions[stopped_instances] = 1
            preds = np.append(preds, predictions)
            sat_labels_shuffled = np.append(sat_labels_shuffled, satisfiability_training_outputs[test_index])
            sat_timeout_labels_shuffled = np.append(sat_timeout_labels_shuffled, sat_timeout_labels[test_index])
            running_time_labels_shuffled = np.append(running_time_labels_shuffled,
                                                     running_time_training_outputs[test_index])
            running_time_labels_timeout_prediction = np.append(running_time_labels_timeout_prediction,
                                                               running_times_timeout_prediction_test)

            fold_eval = eval_timeout_classification_fold(predictions, test_labels_sat, test_running_times,
                                                         feature_collection_cutoff=feature_collection_cutoff)
            metrics[fold] = fold_eval

        eval_final_timeout_classification(predictions=preds, verification_results=sat_labels_shuffled,
                                          timeout_labels=sat_timeout_labels_shuffled,
                                          running_time_labels=running_time_labels_shuffled, metrics=metrics,
                                          threshold=threshold, results_path=results_path_feature,
                                          include_incomplete_results=include_incomplete_results,
                                          feature_collection_cutoff=feature_collection_cutoff,
                                          running_times_timeout_prediction=running_time_labels_timeout_prediction)

def train_continuous_timeout_classifier_feature_ablation_worker(args_queue):
    while True:
        args = args_queue.get()
        if args is None:
            break
        train_continuous_timeout_classifier_feature_ablation(*args)
def run_feature_ablation_study_timeout_classification(config, thresholds=None, results_path=None):
    # TODO: Adjust method description
    """
    Run Timeout prediction experiments using a fixed feature collection phase from a config

    :param config: Refer to sample file experiments/running_time_prediction/config.py
    """

    verification_logs_path = Path(config.get("VERIFICATION_LOGS_PATH", "./verification_logs"))
    experiments = config.get("INCLUDED_EXPERIMENTS", None)
    if not experiments:
        experiments = ALL_EXPERIMENTS

    include_incomplete_results = config.get("INCLUDE_INCOMPLETE_RESULTS", True)

    results_path = './results/feature_ablation/feature_ablation_study/' if not results_path else results_path
    os.makedirs(results_path, exist_ok=True)

    thresholds = config.get("TIMEOUT_CLASSIFICATION_THRESHOLDS", [0.5]) if not thresholds else thresholds

    random_state = config.get("RANDOM_STATE", 42)

    args_queue = multiprocessing.Queue()


    for experiment in experiments:
        print(f"------------------------ Experiment {experiment} -----------------------------")
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
        no_classes = experiment_info.get("no_classes", 10)

        # TODO: REMOVE THAT THATS A HACK!!
        feature_collection_cutoff = 30

        for threshold in thresholds:
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
                        no_classes=no_classes,
                    )
                elif verifier == VERINET:
                    verinet_log_file = os.path.join(experiment_logs_path,
                                                    config.get("VERINET_LOG_NAME", "VERINET.log"))

                    features, running_times, results, enum_results = load_verinet_data(
                        verinet_log_file,
                        neuron_count=experiment_neuron_count,
                        feature_collection_cutoff=feature_collection_cutoff,
                        filter_misclassified=True,
                        no_classes=no_classes
                    )
                elif verifier == OVAL:
                    oval_log_file = os.path.join(experiment_logs_path,
                                                 config.get("OVAL_BAB_LOG_NAME", "OVAL-BAB.log"))

                    features, running_times, results, enum_results = load_oval_bab_data(oval_log_file,
                                                                                        neuron_count=experiment_neuron_count,
                                                                                        feature_collection_cutoff=feature_collection_cutoff,
                                                                                        filter_misclassified=True,
                                                                                        no_classes=no_classes)
                else:
                    # This should never happen!
                    assert 0, "Encountered Unknown Verifier!"
                args_queue.put((features, running_times, verifier_results_path, np.log10(feature_collection_cutoff), enum_results, include_incomplete_results, threshold, random_state, verifier))

    procs = [multiprocessing.Process(target=train_timeout_classifier_feature_ablation_worker, args=(args_queue, ))
             for _ in range(10)]
    for p in procs:
        p.daemon = True
        args_queue.put(None)
        p.start()

    [p.join() for p in procs]



def train_timeout_classifier_feature_ablation_worker(args_queue):
    while True:
        args = args_queue.get()
        if args is None:
            break
        train_timeout_classifier_feature_ablation(*args)


def train_timeout_classifier_feature_ablation(training_inputs, running_times, results_path,
                                              feature_collection_cutoff,
                                              verification_results, include_incomplete_results=False,
                                              threshold=.5, random_state=42, verifier=ABCROWN, ):
    """
    Trains and evaluates a random forest model trained on features collected in a fixed interval
    to predict if an instance will not be solved in a five-fold cross validation.
    :param training_inputs: array of features of all instances
    :param running_times: array of running times of all instances
    :param results_path: path to store results to
    :param feature_collection_cutoff: seconds for which features were collected, i.e. the point in time at
        which the prediction was made
    :param verification_results: array of verification results of all instances
    :param include_incomplete_results: if results solved before feature collection cutoff should be used in training/predictions
    :param threshold: confidence threshold a classification must exceed such that is it counted
    :param random_state: random state for random forest classifier/five-fold cross validation split
    """
    print("---------------------- TRAINING RANDOM FOREST TIMEOUT CLASSIFIER ------------------------")
    if training_inputs is None or running_times is None or verification_results is None:
        print(f"Skipping Experiment for {results_path} - Features or Logs could not be found! ")
        return
    training_inputs = np.array(training_inputs)
    running_time_training_outputs = np.array(running_times)
    satisfiability_training_outputs = np.array(verification_results)

    timeout_indices = np.where(satisfiability_training_outputs == 2)
    no_timeout_indices = np.where(satisfiability_training_outputs != 2)
    sat_timeout_labels = np.copy(satisfiability_training_outputs)
    sat_timeout_labels[timeout_indices] = 1
    sat_timeout_labels[no_timeout_indices] = 0

    if not include_incomplete_results:
        print("--------------------------- EXCLUDING INCOMPLETE RESULTS FROM TRAINING ------------------------------")
        no_incomplete_indices = np.where(running_time_training_outputs > np.log10(10))
        training_inputs = training_inputs[no_incomplete_indices]
        running_time_training_outputs = running_time_training_outputs[no_incomplete_indices]
        satisfiability_training_outputs = satisfiability_training_outputs[no_incomplete_indices]
        sat_timeout_labels = sat_timeout_labels[no_incomplete_indices]

    # Fixed Random State for Comparability between fixed and dynamic Timeout Classification
    kf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)

    split_labels = np.array([1 for _ in verification_results])
    split_labels[running_times < feature_collection_cutoff] = 0
    split_labels[verification_results == 2.] = 2

    VERIFIER_FEATURE_MAP[verifier].append("BASELINE")

    for feature_index, feature in enumerate(VERIFIER_FEATURE_MAP[verifier]):
        print(f"EXCLUDING FEATURE {feature} ON VERIFIER {verifier}")
        results_path_feature = f"{results_path}/{feature}/"
        os.makedirs(results_path_feature, exist_ok=True)

        preds = np.array([])
        sat_labels_shuffled = np.array([])
        running_time_labels_shuffled = np.array([])
        sat_timeout_labels_shuffled = np.array([])
        metrics = {}
        if feature != "BASELINE":
            feature_selector = [x for x in range(training_inputs.shape[1]) if x != feature_index]
        else:
            feature_selector = [x for x in range(training_inputs.shape[1])]
        for fold, (train_index, test_index) in enumerate(kf.split(training_inputs, split_labels)):
            train_inputs = training_inputs[train_index][:, feature_selector]
            test_inputs = training_inputs[test_index][:, feature_selector]
            train_labels_sat = sat_timeout_labels[train_index]
            test_labels_sat = sat_timeout_labels[test_index]

            scaler = StandardScaler().fit(train_inputs)
            train_inputs = scaler.transform(train_inputs)
            test_inputs = scaler.transform(test_inputs)
            test_running_times = running_time_training_outputs[test_index]

            print(f"---------------------------------- Fold {fold} ----------------------------------")

            rf_classifier = RandomForestClassifier(n_estimators=200, random_state=random_state)
            rf_classifier.fit(train_inputs, train_labels_sat)

            probability_predictions = rf_classifier.predict_proba(test_inputs)
            # check needed for case where verifier is extremely sure for all cases
            if probability_predictions.shape[1] > 1:
                y_pred_custom_threshold = (probability_predictions[:, 1] > threshold).astype(int)
                predictions = y_pred_custom_threshold
            else:
                predictions = probability_predictions

            fold_eval = eval_timeout_classification_fold(predictions, test_labels_sat, test_running_times,
                                                         feature_collection_cutoff)

            metrics[fold] = fold_eval

            preds = np.append(preds, predictions)
            sat_labels_shuffled = np.append(sat_labels_shuffled, satisfiability_training_outputs[test_index])
            sat_timeout_labels_shuffled = np.append(sat_timeout_labels_shuffled, sat_timeout_labels[test_index])
            running_time_labels_shuffled = np.append(running_time_labels_shuffled,
                                                     running_time_training_outputs[test_index])

        timeout_running_times = []
        for index, prediction in enumerate(preds):
            if prediction == 1 and running_time_labels_shuffled[index] >= feature_collection_cutoff:
                timeout_running_times.append(feature_collection_cutoff)
            else:
                timeout_running_times.append(running_time_labels_shuffled[index])
        timeout_running_times = np.array(timeout_running_times)
        # shapley_values = np.concatenate(shapley_values, axis=0)

        eval_final_timeout_classification(predictions=preds, verification_results=sat_labels_shuffled,
                                          timeout_labels=sat_timeout_labels_shuffled,
                                          running_time_labels=running_time_labels_shuffled, metrics=metrics,
                                          threshold=threshold, results_path=results_path_feature,
                                          include_incomplete_results=include_incomplete_results,
                                          feature_collection_cutoff=feature_collection_cutoff,
                                          running_times_timeout_prediction=timeout_running_times)


def get_correlated_features(config):
    # TODO: Adjust method description
    """
    Run Timeout prediction experiments using a fixed feature collection phase from a config

    :param config: Refer to sample file experiments/running_time_prediction/config.py
    """

    verification_logs_path = Path(config.get("VERIFICATION_LOGS_PATH", "./verification_logs"))
    experiments = config.get("INCLUDED_EXPERIMENTS", None)
    correlated_features = set()
    if not experiments:
        experiments = ALL_EXPERIMENTS

    for verifier in SUPPORTED_VERIFIERS:
        verifier_features = np.empty(shape=(1, len(VERIFIER_FEATURE_MAP[verifier])))
        for experiment in experiments:
            # skip hidden files
            if experiment.startswith("."):
                continue
            experiment_logs_path = os.path.join(verification_logs_path, experiment)
            experiment_info = config["EXPERIMENTS_INFO"].get(experiment)
            assert experiment_info, f"No Experiment Info for experiment {experiment} provided!"
            experiment_neuron_count = experiment_info.get("neuron_count")
            assert experiment_neuron_count

            feature_collection_cutoff = experiment_info.get("first_classification_at",
                                                            config.get("FEATURE_COLLECTION_CUTOFF", 10))
            feature_collection_cutoff = 30
            if verifier == ABCROWN:
                abcrown_log_file = os.path.join(experiment_logs_path,
                                                config.get("ABCROWN_LOG_NAME", "ABCROWN.log"))
                if not os.path.isfile(abcrown_log_file):
                    continue

                features, running_times, results, enum_results = load_abcrown_data(
                    abcrown_log_file,
                    feature_collection_cutoff=feature_collection_cutoff,
                    neuron_count=experiment_neuron_count
                )
                feature_names = ABCROWN_FEATURE_NAMES
            elif verifier == VERINET:
                verinet_log_file = os.path.join(experiment_logs_path,
                                                config.get("VERINET_LOG_NAME", "VERINET.log"))
                if not os.path.isfile(verinet_log_file):
                    continue
                features, running_times, results, enum_results = load_verinet_data(
                    verinet_log_file,
                    neuron_count=experiment_neuron_count,
                    feature_collection_cutoff=feature_collection_cutoff,
                    filter_misclassified=True
                )
                feature_names = VERINET_FEATURE_NAMES
            elif verifier == OVAL:
                oval_log_file = os.path.join(experiment_logs_path,
                                             config.get("OVAL_BAB_LOG_NAME", "OVAL-BAB.log"))
                if not os.path.isfile(oval_log_file):
                    continue
                features, running_times, results, enum_results = load_oval_bab_data(oval_log_file,
                                                                                    neuron_count=experiment_neuron_count,
                                                                                    feature_collection_cutoff=feature_collection_cutoff,
                                                                                    filter_misclassified=True)
                feature_names = OVAL_FEATURE_NAMES

            unsolved_instances, = np.where(running_times >= np.log10(feature_collection_cutoff))
            verifier_features = np.concatenate((verifier_features, features[unsolved_instances]))
        print(f"----------------- {verifier} -------------------")

        corr_matrix = np.corrcoef(verifier_features, rowvar=False)
        n = corr_matrix.shape[0]
        for i in range(n):
            feature_correlated = False
            for j in range(i + 1, n):
                if abs(corr_matrix[i, j]) > 0.6:
                    print(f"CORRELATED FEATURES: {feature_names[i]}, {feature_names[j]}: {corr_matrix[i, j]}", flush=True)
                    feature_correlated = True
                    correlated_features.add(feature_names[i])
            # if not feature_correlated:
                # print(f"UNCORRELATED FEATURE: {feature_names[i]}")
    # for feature_names in [ABCROWN_FEATURE_NAMES, VERINET_FEATURE_NAMES, OVAL_FEATURE_NAMES]:
    #     print(f"Correlated features: {set.intersection(correlated_features, feature_names)}")
    #     print(f"Uncorrelated features: {set(feature_names) - correlated_features}")
