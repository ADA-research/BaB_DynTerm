
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

import numpy as np

from src.eval.algorithm_selection import eval_fold, eval_final


def adaptive_algorithm_selection(features, best_verifiers, enum_results, verifiers, running_times, frequency,
                                 artificial_cutoff, threshold,
                                 stop_predicted_timeouts=False,
                                 first_classification_at=None, results_path="./results",
                                 random_state=42):
    """
    Function to perform the algorithm selection, including training a classifier at regular checkpoints and
    choosing algorithms accordingly.

    :param features: numpy array of features for each instance. Each feature must be the concatenation of the features of all verifiers.
    :param best_verifiers: numpy array with training labels, i.e., which verifier performed best on each instance.
    :param enum_results: numpy array of verification results for each instance and verifier
    :param verifiers: array of verifiers ran on this experiment
    :param running_times: numpy array of running times for each instance and verifier
    :param frequency: frequency at which algorithm selection is performed
    :param artificial_cutoff: cutoff time, i.e., maximum running time
    :param threshold: confidence threshold a classification must exceed s.t. it is counted
    :param stop_predicted_timeouts: if predicted timeouts should be terminated or if one verification tool should be chosen at random
    :param first_classification_at: seconds after which algorithm should be performed for the first time. If not provided, classification is first performed at 0+frequency
    :param results_path: Path to store results to
    :param random_state: Random state for initialization of Random Forest/Five-Fold split/numpy
    """

    np.random.seed(random_state)
    no_verifiers = len(verifiers)
    fold_evals = {}
    fold_data = {}

    kf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)
    feature_indices = range(len(features[frequency]))

    split_labels = np.array([1 for _ in feature_indices])

    split_cutoff = np.log10(first_classification_at) if first_classification_at is not None else np.log10(frequency)

    for feature_index in feature_indices:
        if np.all(
                running_times[feature_index * no_verifiers:feature_index * no_verifiers + no_verifiers] < split_cutoff):
            split_labels[feature_index] = 0
        elif np.all(enum_results[feature_index * no_verifiers:feature_index * no_verifiers + no_verifiers] == 2.):
            split_labels[feature_index] = 2

    for fold, (train_indices, test_indices) in enumerate(kf.split(feature_indices, split_labels)):
        print(f"---------------------- FOLD {fold} -------------------------------")
        train_best_verifiers = best_verifiers[train_indices]
        test_best_verifiers = best_verifiers[test_indices]
        test_running_times = np.array([])
        test_results = np.array([])
        solved_indices = []
        stopped_indices = []
        ran_with_selected_algo = []
        ran_in_portfolio = []
        selected_verifiers = [None for _ in test_indices]
        selection_running_times = [None for _ in test_indices]
        selection_results = [None for _ in test_indices]

        for test_index in test_indices:
            test_running_times = np.append(
                test_running_times,
                running_times[test_index * no_verifiers:test_index * no_verifiers + no_verifiers]
            )
            test_results = np.append(
                test_results,
                enum_results[test_index * no_verifiers:test_index * no_verifiers + no_verifiers]
            )

        for checkpoint in range(frequency, artificial_cutoff, frequency):

            if first_classification_at and first_classification_at > checkpoint:
                print(f"Skipping algorithm selection at checkpoint {checkpoint} as specified in config!")
                continue

            # TODO Remove
            assert set.intersection(set(stopped_indices), set(solved_indices)) == set(), "Sanity Check failed!"

            if set.union(set(stopped_indices), set(solved_indices)) == set(range(len(test_indices))):
                print("ALL INSTANCES STOPPED OR SOLVED!")
                break

            solved_indices_checkpoint = []
            selected_indices_checkpoint = []
            stopped_indices_checkpoint = []

            train_features = features[checkpoint][train_indices]
            test_features = features[checkpoint][test_indices]

            scaler = preprocessing.StandardScaler().fit(train_features)
            train_features = scaler.transform(train_features)
            test_features = scaler.transform(test_features)

            predictions = get_predictions(train_features=train_features, train_best_verifiers=train_best_verifiers,
                                          test_features=test_features, threshold=threshold)

            for test_index in range(len(test_indices)):
                if test_index in solved_indices or test_index in stopped_indices:
                    continue
                instance_running_times = test_running_times[
                                         test_index * no_verifiers:test_index * no_verifiers + no_verifiers]
                instance_results = test_results[test_index * no_verifiers:test_index * no_verifiers + no_verifiers]

                # instance already solved in portfolio phase!
                if any([running_time < np.log10(checkpoint) for running_time in instance_running_times]):
                    best_runtime = np.min(instance_running_times)
                    best_verifier_index = np.argmin(instance_running_times)
                    selected_verifiers[test_index] = best_verifier_index
                    selection_results[test_index] = instance_results[best_verifier_index]
                    selection_running_times[test_index] = pow(10, best_runtime) * no_verifiers
                    solved_indices_checkpoint.append(test_index)

                    continue

                if first_classification_at > checkpoint:
                    continue

                instance_prediction = predictions[test_index]

                if instance_prediction is None:
                    continue

                if stop_predicted_timeouts and instance_prediction == -1:
                    best_verifier_index = -1
                    instance_result = 2.
                    instance_running_time = checkpoint * no_verifiers
                    stopped_indices_checkpoint.append(test_index)
                    selected_verifiers[test_index] = best_verifier_index
                    selection_results[test_index] = instance_result
                    selection_running_times[test_index] = instance_running_time
                    continue
                elif not stop_predicted_timeouts and instance_prediction == -1:
                    # if predictor says that instance will timeout but we must schedule one verifier,
                    # we choose one at random!
                    best_verifier_index = instance_prediction
                    chosen_verifier_index = np.random.randint(0, no_verifiers)
                    selection_running_time = instance_running_times[chosen_verifier_index]
                    instance_result = instance_results[chosen_verifier_index]
                    selected_indices_checkpoint.append(test_index)
                else:
                    best_verifier_index = instance_prediction
                    selection_running_time = instance_running_times[best_verifier_index]
                    instance_result = instance_results[best_verifier_index]
                    selected_indices_checkpoint.append(test_index)

                # We can calculate the running time here as solved/stopped/undecided indices were stopped earlier
                # calculation: portfolio phase (checkpoint times no. of verifiers) + remaining running time of
                # selected algorithm
                instance_running_time = checkpoint * no_verifiers + (pow(10, selection_running_time) - checkpoint)

                selected_verifiers[test_index] = best_verifier_index
                selection_results[test_index] = instance_result
                selection_running_times[test_index] = instance_running_time

            if solved_indices_checkpoint:
                print(f"SOLVED {solved_indices_checkpoint} at {checkpoint}s!")
            if stopped_indices_checkpoint:
                print(f"STOPPED {stopped_indices_checkpoint} at {checkpoint}s!")
            if selected_indices_checkpoint:
                print(f"SELECTED VERIFIER FOR {selected_indices_checkpoint} at {checkpoint}s!")
            solved_indices = solved_indices + solved_indices_checkpoint + selected_indices_checkpoint
            stopped_indices = stopped_indices + stopped_indices_checkpoint
            ran_in_portfolio = ran_in_portfolio + solved_indices_checkpoint
            ran_with_selected_algo = ran_with_selected_algo + selected_indices_checkpoint + stopped_indices_checkpoint

            if checkpoint == artificial_cutoff - frequency:
                # handle instances for which no verifier has been chosen and all run into timeout
                for test_index in set(range(len(test_indices))) - set(solved_indices) - set(stopped_indices):
                    selection_results[test_index] = 2.
                    selection_running_times[test_index] = artificial_cutoff * no_verifiers
                    selected_verifiers[test_index] = -2
                    ran_in_portfolio = ran_in_portfolio + [test_index]

        fold_eval = eval_fold(
            running_times=test_running_times,
            selection_running_times=selection_running_times,
            results=test_results,
            selection_results=selection_results,
            chosen_verifiers=selected_verifiers,
            verifiers=verifiers,
            vbs_schedules_timeouts=not stop_predicted_timeouts,
            ran_with_selected_algo=ran_with_selected_algo,
            ran_in_portfolio=ran_in_portfolio,
            first_classification_at=first_classification_at if first_classification_at else frequency
        )
        fold_evals[fold] = fold_eval
        fold_data[fold] = {}
        fold_data[fold]["test_running_times"] = test_running_times
        fold_data[fold]["selection_running_times"] = selection_running_times
        fold_data[fold]["results"] = test_results
        fold_data[fold]["selection_results"] = selection_results
        fold_data[fold]["chosen_verifiers"] = selected_verifiers
        fold_data[fold]["best_verifiers"] = test_best_verifiers

    eval_final(
        fold_evals,
        fold_data,
        results_path,
        verifiers=verifiers,
        threshold=threshold
    )


def get_predictions(train_features, train_best_verifiers, test_features, threshold, random_state=42):
    """
    Train a random forest and get best verifier predictions for algorithm selection
    :param train_features: features to train on
    :param train_best_verifiers: labels to train on
    :param test_features: features to classify
    :param threshold: confidence threshold a prediction must exceed such that it is counted
    :param random_state: random state for random forest
    :return: predictions on test features
    """
    predictor = RandomForestClassifier(n_estimators=200, random_state=random_state)
    predictor.fit(train_features, train_best_verifiers)
    if threshold:
        predictions = predictor.predict_proba(test_features)
        if predictions.shape[1] > 1:
            # we substract 1 from the argmax as the valid values are -1 (All will Timeout) and the verifier indices (0-2)
            predictions = [np.argmax(pred) - 1 if np.max(pred) > threshold else None for pred in
                           predictions]
    else:
        predictions = predictor.predict(test_features)

    return predictions

