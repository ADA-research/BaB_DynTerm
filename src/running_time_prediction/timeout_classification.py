import json
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.eval.running_time_prediction import eval_timeout_classification_fold, \
    eval_final_timeout_classification
from src.util.constants import ABCROWN, VERINET, OVAL


def train_timeout_classifier_random_forest(training_inputs, running_times, results_path,
                                           feature_collection_cutoff,
                                           verification_results, include_incomplete_results=False,
                                           threshold=.5, random_state=42):
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
    if training_inputs is None or running_times is None or verification_results is None:
        print(f"Skipping Experiment for {results_path} - Features or Logs could not be found! ")
        return
    print("---------------------- TRAINING RANDOM FOREST TIMEOUT CLASSIFIER ------------------------")

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
        if prediction == 1:
            timeout_running_times.append(feature_collection_cutoff)
        else:
            timeout_running_times.append(running_time_labels_shuffled[index])
    timeout_running_times = np.array(timeout_running_times)

    eval_final_timeout_classification(predictions=preds, verification_results=sat_labels_shuffled,
                                      timeout_labels=sat_timeout_labels_shuffled,
                                      running_time_labels=running_time_labels_shuffled, metrics=metrics,
                                      threshold=threshold, results_path=results_path,
                                      include_incomplete_results=include_incomplete_results,
                                      feature_collection_cutoff=feature_collection_cutoff,
                                      running_times_timeout_prediction=timeout_running_times)


def train_continuous_timeout_classifier(log_path, load_data_func, neuron_count=None, include_incomplete_results=False,
                                        results_path="./results", threshold=.5,
                                        classification_frequency=10, cutoff=600, first_classification_at=None,
                                        no_classes=10, random_state=42):
    """
    Trains and evaluates a timeout prediction model that classifies instances in regular intervals
    :param log_path: path to log file
    :param load_data_func: function to load data from log file
    :param neuron_count: neuron count of evaluated neural network
    :param include_incomplete_results: if instances solved before the first classification should be included in training/prediction
    :param results_path: path to store results to
    :param threshold: confidence threshold a prediction must exceed s.t. it is counted, i.e. s.t. an instance is terminated prematurely
    :param classification_frequency: frequency of checkpoints at which a classification is made
    :param cutoff: running time cutoff of verification runs
    :param first_classification_at: seconds after which the first classification is made, if None perform first classification at
        0+frequency seconds
    :param no_classes: number of output classes in verified neural network
    :param random_state: random state for random forest/five-fold cross validation split
    """

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

    training_times = {}
    preds = np.array([])
    sat_labels_shuffled = np.array([])
    running_time_labels_shuffled = np.array([])
    running_time_labels_timeout_prediction = np.array([])
    sat_timeout_labels_shuffled = np.array([])
    training_inputs_checkpoint = training_inputs[classification_frequency]
    metrics = {}

    feature_collection_cutoff = np.log10(first_classification_at) if first_classification_at else np.log10(
        classification_frequency)

    split_labels = np.array([1 for _ in satisfiability_training_outputs])
    split_labels[running_time_training_outputs < feature_collection_cutoff] = 0
    split_labels[satisfiability_training_outputs == 2.] = 2

    for fold, (train_index, test_index) in enumerate(kf.split(training_inputs_checkpoint, split_labels)):
        train_labels_sat = sat_timeout_labels[train_index]
        test_labels_sat = sat_timeout_labels[test_index]
        running_times_timeout_prediction_test = running_time_training_outputs[test_index].copy()
        test_running_times = running_time_training_outputs[test_index]
        training_times[fold] = {}

        print(f"---------------------------------- Fold {fold} ----------------------------------")

        solved_instances = np.array([], dtype=int)
        stopped_instances = np.array([], dtype=int)

        for checkpoint in range(classification_frequency, cutoff, classification_frequency):
            training_inputs_checkpoint = training_inputs[checkpoint]

            upper_bound_running_time = np.log10(checkpoint)
            lower_bound_running_time = np.log10(checkpoint - classification_frequency)
            solved_instances_checkpoint = np.where(
                (test_running_times >= lower_bound_running_time) & (test_running_times <= upper_bound_running_time) & (
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

            train_inputs = training_inputs_checkpoint[train_index]
            test_inputs = training_inputs_checkpoint[test_index]
            rf_classifier = RandomForestClassifier(n_estimators=200, random_state=random_state)
            rf_train_start = time.perf_counter()
            rf_classifier.fit(train_inputs, train_labels_sat)
            rf_train_end = time.perf_counter()
            rf_training_time = rf_train_end - rf_train_start
            training_times[fold][checkpoint] = rf_training_time

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
                                      threshold=threshold, results_path=results_path,
                                      include_incomplete_results=include_incomplete_results,
                                      feature_collection_cutoff=feature_collection_cutoff,
                                      running_times_timeout_prediction=running_time_labels_timeout_prediction)
    with open(f'{results_path}/training_times_{threshold}.json', 'w', encoding='u8') as f:
        json.dump(training_times, f, indent=2)

def timeout_prediction_baseline(features, running_times, verification_results, verifier,
                                include_incomplete_results=False, results_path="./results",
                                classification_frequency=10, cutoff=600, feature_collection_cutoff=10
                                ):
    """
    Evaluates a baseline on stopping timeouts based on a simple heuristic: if remaining branches times the average time needed
    for the verification of a branch exceeds the maximum running time, predict a timeout
    :param features: array of features of each instance
    :param running_times: array of running times of each instance
    :param verification_results: array of verification results of each instance
    :param verifier: verifier for which the predictions are made
    :param include_incomplete_results: if instances solved before the first checkpoint should be included in training/classification
    :param results_path: path to store results to
    :param classification_frequency: frequency in which instances are classified
    :param cutoff: cutoff time of verification procedure
    :param feature_collection_cutoff: point in time at which first classification is performed
    """
    timeout_indices = np.where(verification_results == 2)
    no_timeout_indices = np.where(verification_results != 2)
    sat_timeout_labels = np.copy(verification_results)
    sat_timeout_labels[timeout_indices] = 1
    sat_timeout_labels[no_timeout_indices] = 0

    if not include_incomplete_results:
        print("--------------------------- EXCLUDING INCOMPLETE RESULTS FROM TRAINING ------------------------------")
        no_incomplete_indices = np.where(running_times > np.log10(feature_collection_cutoff))
        features = features[no_incomplete_indices]
        running_times = running_times[no_incomplete_indices]
        verification_results = verification_results[no_incomplete_indices]
        sat_timeout_labels = sat_timeout_labels[no_incomplete_indices]

    metrics = {}
    solved_instances = np.array([], dtype=int)
    stopped_instances = np.array([], dtype=int)
    running_times_timeout_prediction = running_times.copy()
    for checkpoint in range(classification_frequency, cutoff, classification_frequency):
        features_checkpoint = features[checkpoint]
        upper_bound = np.log10(checkpoint)
        lower_bound = np.log10(checkpoint - classification_frequency)
        solved_instances_checkpoint = np.where(
            (running_times >= lower_bound) & (running_times <= upper_bound) & (
                    verification_results != 2))[0]
        if solved_instances_checkpoint.shape[0] > 0:
            print(f"SOLVED {solved_instances_checkpoint} at {checkpoint} s!")
        solved_instances = np.append(solved_instances, solved_instances_checkpoint)

        if len(running_times) - len(stopped_instances) - len(solved_instances) == 0:
            print("All Test Samples stopped or solved!")
            break

        stopped_instances_checkpoint = []
        for index, instance_features in enumerate(features_checkpoint):
            if index in stopped_instances or index in solved_instances:
                continue

            if verifier == ABCROWN:
                cur_domains, domains_visited = instance_features
            elif verifier == VERINET:
                total_branches, domains_visited = instance_features
                cur_domains = total_branches - domains_visited
            elif verifier == OVAL:
                domains_visited, cur_domains, cur_hard_domains = instance_features
                cur_domains = cur_domains + cur_hard_domains
            else:
                assert False, "Unsupported Verifier!"
            domains_per_second = domains_visited / checkpoint
            time_needed = cur_domains / domains_per_second
            if time_needed + checkpoint > cutoff:
                stopped_instances_checkpoint.append(index)

        print(f"STOPPED INSTANCES {stopped_instances_checkpoint} at {checkpoint}s!")
        stopped_instances = np.append(stopped_instances, np.array(stopped_instances_checkpoint, dtype=int))
        running_times_timeout_prediction[stopped_instances_checkpoint] = np.log10(checkpoint)

    predictions = np.zeros(sat_timeout_labels.shape)
    predictions[stopped_instances] = 1

    fold_eval = eval_timeout_classification_fold(predictions, sat_timeout_labels, test_running_times=running_times,
                                                 feature_collection_cutoff=feature_collection_cutoff)
    metrics[0] = fold_eval

    eval_final_timeout_classification(predictions=predictions, verification_results=verification_results,
                                      timeout_labels=sat_timeout_labels, running_time_labels=running_times,
                                      metrics=metrics, threshold=None, results_path=results_path,
                                      include_incomplete_results=include_incomplete_results,
                                      feature_collection_cutoff=feature_collection_cutoff,
                                      running_times_timeout_prediction=running_times_timeout_prediction)
