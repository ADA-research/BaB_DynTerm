import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.eval.running_time_prediction import eval_running_time_prediction_fold, \
    eval_running_time_prediction_final
from src.util.data_loaders import load_abcrown_data, load_verinet_data, load_oval_bab_data


def train_running_time_predictor_random_forest(training_inputs, running_times,
                                               verification_results, include_timeouts=True,
                                               include_incomplete_results=False, results_path="./results",
                                               feature_collection_cutoff=None, random_state=42):
    print("---------------------- TRAINING RANDOM FOREST RUNNING TIME PREDICTOR ------------------------")
    # load training data
    training_inputs = np.array(training_inputs)
    running_times = np.array(running_times)
    verification_results = np.array(verification_results)
    # training_inputs = preprocessing.normalize(training_inputs, axis=0)

    if not include_timeouts:
        print("--------------------------- EXCLUDING TIMEOUTS FROM TRAINING ------------------------------")
        no_timeout_indices = np.where(verification_results != 2)
        training_inputs = training_inputs[no_timeout_indices]
        running_times = running_times[no_timeout_indices]
        verification_results = verification_results[no_timeout_indices]

    if not include_incomplete_results:
        print("--------------------------- EXCLUDING INCOMPLETE RESULTS FROM TRAINING ------------------------------")
        no_incomplete_indices = np.where(running_times > np.log10(feature_collection_cutoff))
        training_inputs = training_inputs[no_incomplete_indices]
        running_times = running_times[no_incomplete_indices]
        verification_results = verification_results[no_incomplete_indices]

    kf = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)
    predictions = np.array([])
    running_time_labels_shuffled = np.array([])
    sat_labels_shuffled = np.array([])

    metrics = {}
    split_labels = np.array([1 for label in running_times])
    split_labels[running_times < feature_collection_cutoff] = 0
    split_labels[verification_results == 2.] = 2

    for fold, (train_index, test_index) in enumerate(kf.split(training_inputs, split_labels)):
        metrics[fold] = {}
        train_inputs = training_inputs[train_index]
        test_inputs = training_inputs[test_index]
        train_labels_runtime = running_times[train_index]
        test_labels_runtime = running_times[test_index]

        scaler = StandardScaler().fit(train_inputs)
        train_inputs = scaler.transform(train_inputs)
        test_inputs = scaler.transform(test_inputs)

        print(f"---------------------------------- Fold {fold} ----------------------------------")
        rf_regressor = RandomForestRegressor(n_estimators=200, random_state=random_state)
        rf_regressor.fit(X=train_inputs, y=train_labels_runtime)

        fold_metrics = eval_running_time_prediction_fold(rf_regressor=rf_regressor, train_inputs=train_inputs,
                                                         train_labels=train_labels_runtime, test_inputs=test_inputs,
                                                         test_labels=test_labels_runtime,
                                                         feature_collection_cutoff=feature_collection_cutoff,
                                                         test_verification_results=verification_results[test_index])
        metrics[fold] = fold_metrics

        predictions_fold = rf_regressor.predict(test_inputs)

        predictions = np.append(predictions, predictions_fold)
        running_time_labels_shuffled = np.append(running_time_labels_shuffled, test_labels_runtime)
        sat_labels_shuffled = np.append(sat_labels_shuffled, verification_results[test_index])

    eval_running_time_prediction_final(predictions=predictions, running_time_labels=running_time_labels_shuffled,
                                       verification_results=sat_labels_shuffled, results_path=results_path,
                                       feature_collection_cutoff=feature_collection_cutoff, metrics=metrics)