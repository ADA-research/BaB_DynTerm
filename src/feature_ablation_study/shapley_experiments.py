import json
import multiprocessing
import os
import pickle
from pathlib import Path

import numpy as np
import shap.utils
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from experiments.running_time_prediction.config import CONFIG_TIMEOUT_CLASSIFICATION, \
    CONFIG_DYNAMIC_ALGORITHM_TERMINATION
from src.eval.running_time_prediction import eval_final_timeout_classification, eval_timeout_classification_fold
from src.feature_ablation_study.custom_shapley_plots import beeswarm_checkpoint_coloring
from src.feature_ablation_study.shapley_values import get_shapley_explanation
from src.running_time_prediction.timeout_classification import train_timeout_classifier_random_forest
from src.util.constants import SUPPORTED_VERIFIERS, ABCROWN, VERINET, OVAL, ALL_EXPERIMENTS
from src.util.data_loaders import load_verinet_data, load_oval_bab_data, load_abcrown_data

from shap.plots import beeswarm, bar


def get_shapley_values_for_timeout_classification(config):
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

    results_path = './results/feature_ablation/shapley_timeout_classification'
    os.makedirs(results_path, exist_ok=True)

    # WE DO NOT CARE ABOUT THE DIFFERENT THRESHOLDS FOR SHAPLEY VALUES
    thresholds = [0.5]

    random_state = config.get("RANDOM_STATE")

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
                        neuron_count=experiment_neuron_count
                    )
                elif verifier == VERINET:
                    verinet_log_file = os.path.join(experiment_logs_path,
                                                    config.get("VERINET_LOG_NAME", "VERINET.log"))

                    features, running_times, results, enum_results = load_verinet_data(
                        verinet_log_file,
                        neuron_count=experiment_neuron_count,
                        feature_collection_cutoff=feature_collection_cutoff,
                        filter_misclassified=True
                    )
                elif verifier == OVAL:
                    oval_log_file = os.path.join(experiment_logs_path,
                                                 config.get("OVAL_BAB_LOG_NAME", "OVAL-BAB.log"))

                    features, running_times, results, enum_results = load_oval_bab_data(oval_log_file,
                                                                                        neuron_count=experiment_neuron_count,
                                                                                        feature_collection_cutoff=feature_collection_cutoff,
                                                                                        filter_misclassified=True)
                else:
                    # This should never happen!
                    assert 0, "Encountered Unknown Verifier!"

                train_timeout_classifier_with_shapley_explanation(training_inputs=features, running_times=running_times,
                                                                  verification_results=enum_results,
                                                                  threshold=threshold,
                                                                  include_incomplete_results=include_incomplete_results,
                                                                  results_path=verifier_results_path,
                                                                  feature_collection_cutoff=np.log10(
                                                                      feature_collection_cutoff),
                                                                  random_state=random_state,
                                                                  verifier=verifier)


def eval_final_shapley_values(shapley_values_per_fold, results_path):
    plt.rcParams.update({'font.family': 'serif', 'font.serif': 'Times New Roman'})
    for fold, shapley_values in enumerate(shapley_values_per_fold):
        beeswarm(
            shap_values=shapley_values,
            max_display=100,
            plot_size=(25, 25),
            show=False,
            color_bar=True if fold == 0 else False
        )
    plt.savefig(f"{results_path}/final_shapley_values_beeswarm.png")
    plt.close()
    fig, ax = plt.subplots(figsize=(25, 25))
    bar(
        shap_values={f"Fold {fold}": shapley_values for fold, shapley_values in enumerate(shapley_values_per_fold)},
        max_display=100,
        show=False,
        ax=ax
    )
    plt.savefig(f"{results_path}/final_shapley_values_bar.png")
    plt.close()
    with open(f"{results_path}/shapley_values.pkl", "wb") as f:
        pickle.dump(shapley_values_per_fold, f)

def eval_final_shapley_values_continuous_classification(shapley_values_per_fold, results_path):
    plt.rcParams.update({'font.family': 'serif', 'font.serif': 'Times New Roman'})
    beeswarm_checkpoint_coloring(
        shap_values=shapley_values_per_fold,
        max_display=100,
        plot_size=(25, 25),
        show=False,
        color_bar=True,
    )
    plt.savefig(f"{results_path}/final_shapley_values_beeswarm.png")
    plt.close()
    # fig, ax = plt.subplots(figsize=(25, 25))
    # bar(
    #     shap_values={f"Fold {fold}": shapley_values for fold, shapley_values in enumerate(shapley_values_per_fold)},
    #     max_display=100,
    #     show=False,
    #     ax=ax
    # )
    # plt.savefig(f"{results_path}/final_shapley_values_bar.png")
    # plt.close()
    with open(f"{results_path}/shapley_values.pkl", "wb") as f:
        pickle.dump(shapley_values_per_fold, f)


def train_timeout_classifier_with_shapley_explanation(training_inputs, running_times, results_path,
                                                      feature_collection_cutoff,
                                                      verification_results, include_incomplete_results=False,
                                                      threshold=.5, random_state=42, verifier=ABCROWN):
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
    preds = np.array([])
    sat_labels_shuffled = np.array([])
    running_time_labels_shuffled = np.array([])
    sat_timeout_labels_shuffled = np.array([])
    metrics = {}
    shapley_values = []

    split_labels = np.array([1 for _ in verification_results])
    split_labels[running_times < feature_collection_cutoff] = 0
    split_labels[verification_results == 2.] = 2

    for fold, (train_index, test_index) in enumerate(kf.split(training_inputs, split_labels)):
        train_inputs = training_inputs[train_index]
        test_inputs = training_inputs[test_index]
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

        shapley_dict, shapley_values_per_instance = get_shapley_explanation(
            rf_classifier,
            X_train=train_inputs,
            X_test=test_inputs,
            feature_collection_cutoff=feature_collection_cutoff,
            test_running_times=test_running_times,
            results_path=results_path,
            fold=fold
        )

        fold_eval = eval_timeout_classification_fold(predictions, test_labels_sat, test_running_times,
                                                     feature_collection_cutoff)
        for feature_name, shapley_value in shapley_dict.items():
            fold_eval[f"shapley_{feature_name}"] = shapley_value

        metrics[fold] = fold_eval

        preds = np.append(preds, predictions)
        sat_labels_shuffled = np.append(sat_labels_shuffled, satisfiability_training_outputs[test_index])
        sat_timeout_labels_shuffled = np.append(sat_timeout_labels_shuffled, sat_timeout_labels[test_index])
        running_time_labels_shuffled = np.append(running_time_labels_shuffled,
                                                 running_time_training_outputs[test_index])
        shapley_values.append(shapley_values_per_instance)


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
                                      threshold=threshold, results_path=results_path,
                                      include_incomplete_results=include_incomplete_results,
                                      feature_collection_cutoff=feature_collection_cutoff,
                                      running_times_timeout_prediction=timeout_running_times)

    eval_final_shapley_values(shapley_values_per_fold=shapley_values, results_path=results_path)

def train_continuous_timeout_classifier_shapley(log_path, load_data_func, neuron_count=None,
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

    preds = np.array([])
    sat_labels_shuffled = np.array([])
    running_time_labels_shuffled = np.array([])
    sat_timeout_labels_shuffled = np.array([])
    metrics = {}
    running_time_labels_timeout_prediction = np.array([])
    shapley_values = []


    for fold, (train_index, test_index) in enumerate(kf.split(training_inputs[classification_frequency], split_labels)):
        train_labels_sat = sat_timeout_labels[train_index]
        test_labels_sat = sat_timeout_labels[test_index]
        running_times_timeout_prediction_test = running_time_training_outputs[test_index].copy()
        test_running_times = running_time_training_outputs[test_index]

        print(f"---------------------------------- Fold {fold} ----------------------------------")

        solved_instances = np.array([], dtype=int)
        stopped_instances = np.array([], dtype=int)
        checkpoint_shapley_values = []

        for checkpoint in range(classification_frequency, cutoff, classification_frequency):
            training_inputs_checkpoint = training_inputs[checkpoint]

            upper_bound_running_time = np.log10(checkpoint)
            lower_bound_running_time = np.log10(checkpoint - classification_frequency)
            solved_instances_checkpoint = np.where(
                (test_running_times >= lower_bound_running_time) & (
                        test_running_times <= upper_bound_running_time) & (
                        satisfiability_training_outputs[test_index] != 2))[0]
            # if solved_instances_checkpoint.shape[0] > 0:
                # print(f"SOLVED {solved_instances_checkpoint} at {checkpoint} s!")
            solved_instances = np.append(solved_instances, solved_instances_checkpoint)

            if len(test_index) - len(stopped_instances) - len(solved_instances) == 0:
                # print("All Test Samples stopped or solved!")
                break

            if first_classification_at and checkpoint < first_classification_at:
                # print(f"Skipping Timeout Prediction at checkpoint {checkpoint} as specified in config!")
                continue

            train_inputs = training_inputs_checkpoint[train_index]
            test_inputs = training_inputs_checkpoint[test_index]
            rf_classifier = RandomForestClassifier(n_estimators=200, random_state=random_state)
            rf_classifier.fit(train_inputs, train_labels_sat)

            probability_predictions = rf_classifier.predict_proba(test_inputs)
            if probability_predictions.shape[1] > 1:
                y_pred_custom_threshold = (probability_predictions[:, 1] > threshold).astype(int)
                predictions = y_pred_custom_threshold
            else:
                predictions = probability_predictions

            shapley_dict, shapley_values_per_instance = get_shapley_explanation(
                rf_classifier,
                X_train=training_inputs_checkpoint,
                X_test=test_inputs,
                feature_collection_cutoff=np.log10(checkpoint),
                test_running_times=test_running_times,
                results_path=results_path,
                fold=f"{fold}_checkpoint_{checkpoint}",
                checkpoint=checkpoint
            )
            checkpoint_shapley_values.append(shapley_values_per_instance)

            stopped_instances_checkpoint = []
            for index, test_sample in enumerate(test_index):
                if predictions[index] == 1 and not np.isin(index, stopped_instances) and not np.isin(index,
                                                                                                     solved_instances):
                    stopped_instances_checkpoint.append(index)
            # print(f"STOPPED INSTANCES {stopped_instances_checkpoint} at {checkpoint}s!")
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
        shapley_values.append(checkpoint_shapley_values)

    eval_final_timeout_classification(predictions=preds, verification_results=sat_labels_shuffled,
                                      timeout_labels=sat_timeout_labels_shuffled,
                                      running_time_labels=running_time_labels_shuffled, metrics=metrics,
                                      threshold=threshold, results_path=results_path,
                                      include_incomplete_results=include_incomplete_results,
                                      feature_collection_cutoff=feature_collection_cutoff,
                                      running_times_timeout_prediction=running_time_labels_timeout_prediction)
    eval_final_shapley_values_continuous_classification(
        shapley_values_per_fold=shapley_values,
        results_path=results_path,
    )


def train_continuous_timeout_classifier_shapley_worker(args_queue):
    while True:
        args = args_queue.get()
        if args is None:
            break
        train_continuous_timeout_classifier_shapley(*args)
