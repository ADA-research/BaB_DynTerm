import os
import pickle

import numpy as np
import sklearn.model_selection
from sklearn.model_selection import StratifiedKFold
import autosklearn.regression
import autosklearn.metrics

from src.eval.running_time_prediction import eval_running_time_prediction_fold, \
    eval_running_time_prediction_final
from src.util.data_loaders import load_verinet_data, load_abcrown_data, load_oval_bab_data


def auto_train_running_time_predictor_random_forest(training_inputs, running_times,
                                                    verification_results, include_timeouts=True,
                                                    include_incomplete_results=False, results_path="./results",
                                                    feature_collection_cutoff=None):
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

    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    predictions = np.array([])
    running_time_labels_shuffled = np.array([])
    sat_labels_shuffled = np.array([])

    metrics = {}

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=7200,
        per_run_time_limit=300,
        metric=autosklearn.metrics.root_mean_squared_error,
        include={"regressor": ["random_forest"]},
        resampling_strategy="cv",
        resampling_strategy_arguments={"folds": 5},
        ensemble_kwargs={"ensemble_size": 1}
    )
    split_labels = np.array([1 for _ in running_times])
    split_labels[running_times < feature_collection_cutoff] = 0
    split_labels[verification_results == 2.] = 2
    X_train, _, y_train, _ = sklearn.model_selection.train_test_split(training_inputs, running_times, stratify=split_labels)
    automl.fit(X_train, y_train)
    with open(f"{results_path}/cv_results.pkl", "wb") as f:
        pickle.dump(automl.cv_results_, f)
    with open(f"{results_path}/performance_over_time.pkl", "wb") as f:
        pickle.dump(automl.performance_over_time_, f)

    print(automl.leaderboard())
    print("\n")
    print(automl.show_models())
    with open(f"{results_path}/model.pkl", 'wb') as f:
        pickle.dump(automl, f)

    for fold, (train_index, test_index) in enumerate(kf.split(training_inputs, split_labels)):
        metrics[fold] = {}
        train_inputs = training_inputs[train_index]
        test_inputs = training_inputs[test_index]
        train_labels_runtime = running_times[train_index]
        test_labels_runtime = running_times[test_index]

        # scaler = StandardScaler().fit(train_inputs)
        # train_inputs = scaler.transform(train_inputs)
        # test_inputs = scaler.transform(test_inputs)

        print(f"---------------------------------- Fold {fold} ----------------------------------")

        automl.refit(X=train_inputs, y=train_labels_runtime)

        fold_metrics = eval_running_time_prediction_fold(
            rf_regressor=automl,
            train_inputs=train_inputs,
            train_labels=train_labels_runtime,
            test_inputs=test_inputs,
            test_labels=test_labels_runtime,
            feature_collection_cutoff=np.log10(feature_collection_cutoff),
            sat_labels=verification_results[test_index]
        )
        metrics[fold] = fold_metrics

        predictions_fold = automl.predict(test_inputs)

        predictions = np.append(predictions, predictions_fold)
        running_time_labels_shuffled = np.append(running_time_labels_shuffled, test_labels_runtime)
        sat_labels_shuffled = np.append(sat_labels_shuffled, verification_results[test_index])

    eval_running_time_prediction_final(
        predictions=predictions,
        running_time_labels=running_time_labels_shuffled,
        sat_labels=sat_labels_shuffled,
        results_path=results_path,
        feature_collection_cutoff=np.log10(feature_collection_cutoff),
        metrics=metrics
    )


if __name__ == "__main__":
    results_path = "./autosklearn_only_random_forest"
    os.makedirs(results_path, exist_ok=True)
    verinet_log_file = f"verification_logs/OVAL21/VERINET.log"
    features, running_times, results, enum_results = load_verinet_data(
        verinet_log_file,
        neuron_count=8519,
        feature_collection_cutoff=10,
        filter_misclassified=True,
        artificial_cutoff=600
    )
    os.makedirs(f"{results_path}/VERINET", exist_ok=True)
    auto_train_running_time_predictor_random_forest(features, running_times, enum_results,
                                                    include_incomplete_results=True, feature_collection_cutoff=10,
                                                    results_path=f"{results_path}/VERINET")
    abcrown_log_file = f"verification_logs/OVAL21/abCROWN.log"
    features, running_times, results, enum_results = load_abcrown_data(
        abcrown_log_file,
        neuron_count=8519,
        feature_collection_cutoff=10,
        filter_misclassified=True,
        artificial_cutoff=600
    )
    os.makedirs(f"{results_path}/ABCROWN", exist_ok=True)
    auto_train_running_time_predictor_random_forest(features, running_times, enum_results,
                                                    include_incomplete_results=True, feature_collection_cutoff=10,
                                                    results_path=f"{results_path}/ABCROWN")
    oval_log_file = f"verification_logs/OVAL21/OVAL-BAB.log"
    features, running_times, results, enum_results = load_oval_bab_data(
        oval_log_file,
        neuron_count=8519,
        feature_collection_cutoff=10,
        filter_misclassified=True,
        artificial_cutoff=600
    )
    os.makedirs(f"{results_path}/OVAL_BAB", exist_ok=True)
    auto_train_running_time_predictor_random_forest(features, running_times, enum_results,
                                                    include_incomplete_results=True, feature_collection_cutoff=10,
                                                    results_path=f"{results_path}/OVAL_BAB")
