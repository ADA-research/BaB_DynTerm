from itertools import combinations

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold

import numpy as np

from src.eval.algorithm_selection import eval_fold, eval_final
from src.util.constants import TIMEOUT


def algorithm_selection_running_time_regression(features, running_times, enum_results, verifiers,
                                                feature_collection_cutoff=None, stop_predicted_timeouts=False):
    no_verifiers = len(verifiers)
    indices = np.array(range(int(len(running_times) / no_verifiers)))
    max_running_time = pow(10, max(running_times))
    fold_evals = {}

    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    split_labels = np.array([1 for _ in running_times])
    split_labels[running_times < feature_collection_cutoff] = 0
    split_labels[enum_results == 2.] = 2

    for fold, (train_indices, test_indices) in enumerate(kf.split(features, split_labels)):
        print(f"---------------------- FOLD {fold} -------------------------------")

        train_indices = indices[train_indices]
        test_indices = indices[test_indices]

        train_features = np.array([])
        train_running_times = np.array([])
        for train_index in train_indices:
            train_features = np.append(
                train_features,
                features[train_index * no_verifiers:train_index * no_verifiers + no_verifiers]
            )
            train_running_times = np.append(
                train_running_times,
                running_times[train_index * no_verifiers:train_index * no_verifiers + no_verifiers]
            )
        train_features = np.reshape(train_features, (len(train_indices) * no_verifiers, -1))

        test_features = np.array([])
        test_running_times = np.array([])
        test_results = np.array([])
        for test_index in test_indices:
            test_features = np.append(
                test_features,
                features[test_index * no_verifiers:test_index * no_verifiers + no_verifiers]
            )
            test_running_times = np.append(
                test_running_times,
                running_times[test_index * no_verifiers:test_index * no_verifiers + no_verifiers]
            )
            test_results = np.append(
                test_results,
                enum_results[test_index * no_verifiers:test_index * no_verifiers + no_verifiers]
            )
        test_features = np.reshape(test_features, (len(test_indices) * no_verifiers, -1))

        scaler = preprocessing.StandardScaler().fit(train_features)
        train_features = scaler.transform(train_features)
        test_features = scaler.transform(test_features)

        predictor = RandomForestRegressor(n_estimators=200, random_state=42)
        predictor.fit(train_features, train_running_times)

        selected_verifiers = []
        selection_running_times = []
        selection_results = []
        for test_index in range(len(test_indices)):
            instance_features = test_features[test_index * no_verifiers:test_index * no_verifiers + no_verifiers]
            instance_running_times = test_running_times[
                                     test_index * no_verifiers:test_index * no_verifiers + no_verifiers]
            instance_results = test_results[test_index * no_verifiers:test_index * no_verifiers + no_verifiers]
            instance_running_time = 0

            # instance already solved during feature collection --> skip with best running time times portfolio overhead!
            if any([running_time < np.log10(feature_collection_cutoff) for running_time in instance_running_times]):
                best_runtime = np.min(instance_running_times)
                best_verifier_index = np.argmin(instance_running_times)
                selected_verifiers.append(best_verifier_index)
                selection_results.append(instance_results[best_verifier_index])
                selection_running_times.append(pow(10, best_runtime) * no_verifiers)
                continue
            else:
                instance_running_time += feature_collection_cutoff * no_verifiers

            prediction = predictor.predict(instance_features)

            if stop_predicted_timeouts and all(
                    [max_running_time - 10 <= pow(10, pred) <= max_running_time for pred in prediction]):
                best_verifier_index = -1
                instance_result = 2.
            else:
                best_verifier_index = np.argmin(prediction)
                selection_running_time = instance_running_times[best_verifier_index]
                instance_result = instance_results[best_verifier_index]
                instance_running_time += (pow(10, selection_running_time) - feature_collection_cutoff)

            selected_verifiers.append(best_verifier_index)
            selection_results.append(instance_result)
            selection_running_times.append(instance_running_time)

        fold_eval = eval_fold(
            running_times=test_running_times,
            selection_running_times=selection_running_times,
            results=test_results,
            selection_results=selection_results,
            chosen_verifiers=selected_verifiers,
            verifiers=verifiers,
            vbs_schedules_timeouts=not stop_predicted_timeouts,
            first_classification_at=feature_collection_cutoff
        )
        fold_evals[fold] = fold_eval


def algorithm_selection_classification(features, best_verifiers, enum_results, verifiers, running_times,
                                       feature_collection_cutoff=None, stop_predicted_timeouts=False):
    no_verifiers = len(verifiers)
    fold_evals = {}

    kf = KFold(n_splits=5, random_state=None, shuffle=True)

    split_labels = np.array([1 for _ in running_times])
    split_labels[running_times < feature_collection_cutoff] = 0
    split_labels[enum_results == 2.] = 2

    for fold, (train_indices, test_indices) in enumerate(kf.split(features, split_labels)):

        print(f"---------------------- FOLD {fold} -------------------------------")

        train_features = features[train_indices]
        train_best_verifiers = best_verifiers[train_indices]
        test_features = features[test_indices]
        test_best_verifiers = best_verifiers[test_indices]

        test_running_times = np.array([])
        test_results = np.array([])
        for test_index in test_indices:
            test_running_times = np.append(
                test_running_times,
                running_times[test_index * no_verifiers:test_index * no_verifiers + no_verifiers]
            )
            test_results = np.append(
                test_results,
                enum_results[test_index * no_verifiers:test_index * no_verifiers + no_verifiers]
            )

        scaler = preprocessing.StandardScaler().fit(train_features)
        train_features = scaler.transform(train_features)
        test_features = scaler.transform(test_features)

        predictor = RandomForestClassifier(n_estimators=200, random_state=42)
        predictor.fit(train_features, train_best_verifiers)
        predictions = predictor.predict(test_features)

        selected_verifiers = []
        selection_running_times = []
        selection_results = []
        for test_index in range(len(test_indices)):
            instance_running_times = test_running_times[
                                     test_index * no_verifiers:test_index * no_verifiers + no_verifiers]
            instance_results = test_results[test_index * no_verifiers:test_index * no_verifiers + no_verifiers]
            instance_running_time = 0

            # instance already solved during feature collection --> skip with best running time times portfolio overhead!
            if any([running_time < np.log10(feature_collection_cutoff) for running_time in instance_running_times]):
                best_runtime = np.min(instance_running_times)
                best_verifier_index = np.argmin(instance_running_times)
                selected_verifiers.append(best_verifier_index)
                selection_results.append(instance_results[best_verifier_index])
                selection_running_times.append(pow(10, best_runtime) * no_verifiers)
                continue
            else:
                instance_running_time += feature_collection_cutoff * no_verifiers

            instance_prediction = predictions[test_index]

            if stop_predicted_timeouts and predictions[test_index] == -1:
                best_verifier_index = -1
                instance_result = 2.
            elif not stop_predicted_timeouts and predictions[test_index] == -1:
                # if predictor says that instance will timeout but we must schedule one verifier,
                # we choose one at random!
                best_verifier_index = instance_prediction
                chosen_verifier_index = np.random.randint(0, no_verifiers)
                selection_running_time = instance_running_times[chosen_verifier_index]
                instance_result = instance_results[chosen_verifier_index]
                instance_running_time += pow(10, selection_running_time)
            else:
                best_verifier_index = instance_prediction
                selection_running_time = instance_running_times[best_verifier_index]
                instance_result = instance_results[best_verifier_index]
                instance_running_time += (pow(10, selection_running_time) - feature_collection_cutoff)

            selected_verifiers.append(best_verifier_index)
            selection_results.append(instance_result)
            selection_running_times.append(instance_running_time)

        fold_eval = eval_fold(
            running_times=test_running_times,
            selection_running_times=selection_running_times,
            results=test_results,
            selection_results=selection_results,
            chosen_verifiers=selected_verifiers,
            verifiers=verifiers,
            vbs_schedules_timeouts=not stop_predicted_timeouts,
            first_classification_at=feature_collection_cutoff
        )
        fold_evals[fold] = fold_eval


def adaptive_algorithm_selection(features, best_verifiers, enum_results, verifiers, running_times, frequency,
                                 verifier_data, artificial_cutoff, threshold,
                                 feature_collection_cutoff=None, stop_predicted_timeouts=False,
                                 classification_method="NAIVE",
                                 first_classification_at=None, results_path="./results"):
    no_verifiers = len(verifiers)
    fold_evals = {}
    fold_data = {}

    # TODO: Add Presolving Phase where SBS (abCROWN) runs for X seconds to filter out "low hanging fruits" (?)

    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
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

            if classification_method == "NAIVE":
                predictions = get_naive_predictions(
                    train_features=train_features,
                    train_best_verifiers=train_best_verifiers,
                    threshold=threshold,
                    test_features=test_features
                )
            elif classification_method == "PAIRWISE":
                predictions = get_pairwise_predictions(
                    train_indices=train_indices,
                    test_indices=test_indices,
                    test_features=test_features,
                    checkpoint=checkpoint,
                    threshold=threshold,
                    verifier_data=verifier_data
                )
            else:
                assert False, "Unsupported Classification Method!"

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

                    if best_verifier_index != 0 and checkpoint > 10 and instance_running_times[0] > np.log10(30):
                        print(f"INSTANCE {test_index} SOLVED IN PORTFOLIO!")
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

        # TODO DELETE THAT
        print("RAN IN PORTFOLIO", ran_in_portfolio)
        print("SELECTED", ran_with_selected_algo)


    eval_final(
        fold_evals,
        fold_data,
        results_path,
        verifiers=verifiers,
        threshold=threshold
    )


def get_naive_predictions(train_features, train_best_verifiers, test_features, threshold):
    predictor = RandomForestClassifier(n_estimators=200, random_state=42)
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


def get_pairwise_predictions(train_indices, test_indices, test_features, checkpoint, verifier_data, threshold):
    supported_verifiers = verifier_data.keys()
    pairwise_combinations = combinations(supported_verifiers, 2)
    pairwise_classifiers = {}
    for combination in pairwise_combinations:
        verifier_1, verifier_2 = combination
        train_features = np.concatenate(
            (verifier_data[verifier_1]["features"][checkpoint][train_indices],
             verifier_data[verifier_2]["features"][checkpoint][train_indices]),
            axis=1)
        train_labels = verifier_data[verifier_1]["running_times"][train_indices] > \
                       verifier_data[verifier_2]["running_times"][train_indices]
        train_labels = train_labels.astype(int)
        both_timeout_indices = (verifier_data[verifier_1]["enum_results"][train_indices] == TIMEOUT) & (
                verifier_data[verifier_1]["enum_results"][train_indices] == TIMEOUT)
        train_labels[both_timeout_indices] = -1
        predictor = RandomForestClassifier(n_estimators=200, random_state=42)

        predictor.fit(train_features, train_labels)
        pairwise_classifiers[combination] = predictor

    votes = np.zeros((len(test_indices), len(supported_verifiers) + 1))
    verifier_positions = {
        verifier: position + 1    # we take position + 1 as index 0 is reserved for timeout
        for (position, verifier) in enumerate(supported_verifiers)
    }
    uncertain_indices = []
    for combination in combinations(supported_verifiers, 2):
        verifier_1, verifier_2 = combination
        test_features = np.concatenate(
            (verifier_data[verifier_1]["features"][checkpoint][test_indices],
             verifier_data[verifier_2]["features"][checkpoint][test_indices]),
            axis=1)
        predictor = pairwise_classifiers[combination]

        if threshold:
            pairwise_predictions = predictor.predict_proba(test_features)
            if pairwise_predictions.shape[1] > 1:
                pairwise_predictions = [np.argmax(pred) if np.max(pred) > threshold else None for pred in
                                        pairwise_predictions]
        else:
            pairwise_predictions = predictor.predict(test_features)

        for index, pred in enumerate(pairwise_predictions):
            if pred is None:
                # if any predictor did not surpass the confidence threshold, we postpone classification!
                uncertain_indices.append(index)
                uncertain_indices = list(set(uncertain_indices))
            elif pred == 0:
                votes[index][0] += 1
                if index in uncertain_indices:
                    uncertain_indices.remove(index)
            else:
                pred_verifier = combination[pred - 1]
                pred_verifier_index = verifier_positions[pred_verifier]
                votes[index][pred_verifier_index] += 1

                if index in uncertain_indices:
                    uncertain_indices.remove(index)

    # TODO: Weitersammeln bis man eine eindeutige Entscheidung hat!
    # TODO: Erst stoppen wenn ALLE verifier sagen bitte stoppen!
    predictions = np.argmax(votes, axis=1)
    # substract 1 from whole array to conform with class labels (-1): all timeout, (0-2): verifier indices
    predictions = predictions - 1
    # set uncertain indices to None
    predictions = predictions.tolist()
    for index in uncertain_indices:
        predictions[index] = None
    return predictions
