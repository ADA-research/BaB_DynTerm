import json
import math
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, \
    r2_score, precision_score, recall_score, f1_score, fbeta_score
from scipy.stats import spearmanr

from src.util.visualization.diagrams import create_scatter_plot, create_confusion_matrix, create_ecdf_plot
from src.util.constants import TIMEOUT


def eval_running_time_prediction_fold(rf_regressor, train_inputs, train_labels, test_inputs, test_labels,
                                      feature_collection_cutoff, test_verification_results):
    """
    Function to evaluate a fold of running time regression predictions according to several metrics
    :param rf_regressor: trained sklearn random forest regression model
    :param train_inputs: features the regression model was trained on
    :param train_labels: labels the regression model was trained on
    :param test_inputs: inputs from test set to evaluate the model on
    :param test_labels: labels from test set to evaluate the model on
    :param feature_collection_cutoff: seconds for which features were collected, i.e. the point in time at which the prediction was made
    :param test_verification_results: verification results of the test set instances
    :return: dict of metrics of the fold
    """

    metrics = {}

    score = rf_regressor.score(train_inputs, train_labels)
    metrics["r2_train"] = score

    score = rf_regressor.score(test_inputs, test_labels)
    metrics["r2_test"] = score

    train_predictions = rf_regressor.predict(train_inputs)

    mse_train = mean_squared_error(train_labels, train_predictions, squared=False)
    metrics["rmse_train"] = mse_train

    mae_train = mean_absolute_error(train_labels, train_predictions)
    metrics["mae_train"] = mae_train

    x_left = np.where(train_labels <= feature_collection_cutoff)[0]
    x_right = np.where(train_labels > feature_collection_cutoff)[0]

    if x_left.shape[0] > 0 and x_right.shape[0] > 0:
        rmse_train_left = mean_squared_error(train_labels[x_left], train_predictions[x_left], squared=False)
        rmse_train_right = mean_squared_error(train_labels[x_right], train_predictions[x_right], squared=False)
        mae_train_left = mean_absolute_error(train_labels[x_left], train_predictions[x_left])
        mae_train_right = mean_absolute_error(train_labels[x_right], train_predictions[x_right])

        metrics["rmse_train_left"] = rmse_train_left
        metrics["rmse_train_right"] = rmse_train_right
        metrics["mae_train_left"] = mae_train_left
        metrics["mae_train_right"] = mae_train_right

    spearman_rank_correlation_train = spearmanr(train_labels, train_predictions)
    metrics["spearman_correlation_train"] = spearman_rank_correlation_train.correlation
    metrics["spearman_correlation_pvalue_train"] = spearman_rank_correlation_train.pvalue

    test_predictions = rf_regressor.predict(test_inputs)

    rmse_val = mean_squared_error(test_labels, test_predictions, squared=False)
    metrics["rmse_test"] = rmse_val

    mae_test = mean_absolute_error(test_labels, test_predictions)
    metrics["mae_test"] = mae_test

    x_left = np.where(test_labels <= feature_collection_cutoff)[0]
    x_right = np.where(test_labels > feature_collection_cutoff)[0]

    max_running_time_train = max(train_labels)
    timeout_cutoff = np.log10(pow(10, max_running_time_train) - 10)
    timeout_predictions = np.where(test_predictions > timeout_cutoff, 1, 0)
    timeouts = np.where(test_verification_results == TIMEOUT, 1, 0)
    timeout_acc = accuracy_score(timeouts, timeout_predictions)
    timeout_precision = precision_score(timeouts, timeout_predictions)
    timeout_recall = recall_score(timeouts, timeout_predictions)
    timeout_f1 = f1_score(timeouts, timeout_predictions)
    metrics["timeout_acc"] = timeout_acc
    metrics["timeout_precision"] = timeout_precision
    metrics["timeout_recall"] = timeout_recall
    metrics["timeout_f1"] = timeout_f1

    spearman_rank_correlation_test = spearmanr(test_labels, test_predictions)
    metrics["spearman_correlation_test"] = spearman_rank_correlation_test.correlation
    metrics["spearman_correlation_pvalue_test"] = spearman_rank_correlation_test.pvalue

    if x_left.shape[0] > 0 and x_right.shape[0] > 0:
        rmse_test_left = mean_squared_error(test_labels[x_left], test_predictions[x_left], squared=False)
        rmse_test_right = mean_squared_error(test_labels[x_right], test_predictions[x_right], squared=False)
        mae_test_left = mean_absolute_error(test_labels[x_left], test_predictions[x_left])
        mae_test_right = mean_absolute_error(test_labels[x_right], test_predictions[x_right])

        metrics["rmse_test_left"] = rmse_test_left
        metrics["rmse_test_right"] = rmse_test_right
        metrics["mae_test_left"] = mae_test_left
        metrics["mae_test_right"] = mae_test_right

        spearman_rank_correlation_test_left = spearmanr(test_labels[x_left], test_predictions[x_left])
        spearman_rank_correlation_test_right = spearmanr(test_labels[x_right], test_predictions[x_right])

        metrics["spearman_correlation_test_left"] = spearman_rank_correlation_test_left.correlation
        metrics["spearman_correlation_pvalue_test_left"] = spearman_rank_correlation_test_left.pvalue
        metrics["spearman_correlation_test_right"] = spearman_rank_correlation_test_right.correlation
        metrics["spearman_correlation_pvalue_test_right"] = spearman_rank_correlation_test_right.pvalue

        r2_score_test_left = r2_score(test_labels[x_left], test_predictions[x_left])
        r2_score_test_right = r2_score(test_labels[x_right], test_predictions[x_right])
        metrics["r2_test_left"] = r2_score_test_left
        metrics["r2_test_right"] = r2_score_test_right

    return metrics


def eval_running_time_prediction_final(predictions, running_time_labels, verification_results, results_path,
                                       feature_collection_cutoff, metrics):
    """
    Function to aggregate all running time regression fold evaluations and to create scatter plot of all test set predictions
    :param predictions: concatenated test set predictions of all folds
    :param running_time_labels: concatenated true running times of test sets of all folds
    :param verification_results: concatenated verification results of test sets of all folds
    :param results_path: path to store results to
    :param feature_collection_cutoff: seconds for which features were collected, i.e. the point in time at which the prediction was made
    :param metrics: dict of fold evaluations of all folds
    """

    for legend, filename in [("full", os.path.join(results_path, "scatter_plot_with_legend.pdf")),
                             (False, os.path.join(results_path, "scatter_plot.pdf"))]:
        create_scatter_plot(predictions, running_time_labels, satisfiability_labels=verification_results,
                            filename=filename,
                            feature_collection_cutoff=feature_collection_cutoff,
                            legend=legend)

    avg_metrics = {}
    for fold, fold_metrics in metrics.items():
        for metric_name, metric_value in fold_metrics.items():
            if math.isnan(metric_value):
                continue
            avg_metrics[metric_name] = avg_metrics.get(metric_name, 0) + metric_value

    avg_metrics = {metric_name: total_value / len(metrics) for metric_name, total_value in avg_metrics.items()}
    metrics["avg"] = avg_metrics

    metrics_file_path = os.path.join(results_path, "running_time_prediction_metrics.json")
    with open(metrics_file_path, "w", encoding='u8') as f:
        json.dump(metrics, f, indent=2)


def eval_timeout_classification_fold(test_predictions, test_labels, test_running_times,
                                     feature_collection_cutoff):
    """
    Function to evaluate a fold of timeout predictions according to several metrics
    :param test_predictions: timeout predictions on test set
    :param test_labels: true timeout labels of test set instances
    :param test_running_times: running times of test set instances
    :param feature_collection_cutoff: seconds for which features were collected, i.e. the point in time at which the prediction was made
    :return: dict of metrics of the fold
    """

    unsolved_instances = np.where(test_running_times > feature_collection_cutoff)[0]
    if len(unsolved_instances) == 0:
        return None
    test_labels = test_labels[unsolved_instances]
    test_predictions = test_predictions[unsolved_instances]
    test_acc = accuracy_score(test_labels, test_predictions)
    print("ACC ON TEST SET:", test_acc)

    test_confusion_matrix = confusion_matrix(test_predictions, test_labels)

    # this is a terrible edge case that outputs a 1x1 confusion matrix if there
    # is only one class in predictions and true labels
    if test_confusion_matrix.shape != (2, 2):
        no_instances = len(test_predictions)
        if all([pred == 1 for pred in test_predictions]):
            true_positives, true_negatives, false_positives, false_negatives, tpr, tnr, fpr, fnr = \
                no_instances, 0, 0, 0, 1.0, float("nan"), float("nan"), 0
        elif all([pred == 0 for pred in test_predictions]):
            true_positives, true_negatives, false_positives, false_negatives, tpr, tnr, fpr, fnr = \
                0, no_instances, 0, 0, float("nan"), 1.0, 0, float("nan")
        else:
            assert False, "Encountered Unknown Confusion Matrix Shape"
    else:
        true_positives = test_confusion_matrix[1][1]
        true_negatives = test_confusion_matrix[0][0]
        false_positives = test_confusion_matrix[1][0]
        false_negatives = test_confusion_matrix[0][1]
        tpr = true_positives / (true_positives + false_negatives)
        tnr = true_negatives / (true_negatives + false_positives)
        fpr = false_positives / (false_positives + true_negatives)
        fnr = false_negatives / (false_negatives + true_positives)

    metrics = {
        "test_acc": float(test_acc),
        "tp": int(true_positives),
        "tn": int(true_negatives),
        "fp": int(false_positives),
        "fn": int(false_negatives),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "tnr": float(tnr),
        "fnr": float(fnr),
        "f1_score": f1_score(test_labels, test_predictions),
        "f0.5_score": fbeta_score(test_labels, test_predictions, beta=0.5),
        "f2_score": fbeta_score(test_labels, test_predictions, beta=2),
        "precision": precision_score(test_labels, test_predictions),
        "recall": recall_score(test_labels, test_predictions)
    }

    return metrics


def eval_final_timeout_classification(predictions, verification_results, timeout_labels, running_time_labels, metrics, threshold,
                                      results_path, include_incomplete_results, feature_collection_cutoff,
                                      running_times_timeout_prediction):
    """
    Function to aggregate all timeout prediction fold evaluations
    and to create scatter + ECDF plot and confusion matrix of all test set predictions

    :param predictions: concatenation of all test set predictions over all folds
    :param verification_results: concatenation of all true test set verification results over all folds
    :param timeout_labels: concatenation of all timeout labels over all folds
    :param running_time_labels: concatenation of all true running times over all folds
    :param metrics: dict with metrics of each fold
    :param threshold: confidence threshold a prediction must exceed such that it is counted
    :param results_path: path to store results to
    :param include_incomplete_results: if instances solved before feature collection cutoff should be included in scatter plot/confusion matrix
    :param feature_collection_cutoff: seconds for which features are collected and after which the predictions are made
    :param running_times_timeout_prediction: resulting running times if instances were stopped once they are predicted as timeouts
    """

    filename_confusion_matrix = os.path.join(results_path, f"confusion_matrix_threshold_{threshold}.png")
    filename_scatter_plot = os.path.join(results_path, f"scatter_timeout_classification_threshold_{threshold}.png")

    avg_metrics = {}
    sum_metrics = defaultdict(float)
    no_folds = len(metrics)

    for fold, fold_metrics in metrics.items():
        sum_metrics["tp"] += fold_metrics["tp"]
        sum_metrics["fp"] += fold_metrics["fp"]
        sum_metrics["tn"] += fold_metrics["tn"]
        sum_metrics["fn"] += fold_metrics["fn"]

    try:
        sum_metrics["tpr"] = sum_metrics["tp"] / (sum_metrics["tp"] + sum_metrics["fn"])
        sum_metrics["tnr"] = sum_metrics["tn"] / (sum_metrics["tn"] + sum_metrics["fp"])
        sum_metrics["fpr"] = sum_metrics["fp"] / (sum_metrics["fp"] + sum_metrics["tn"])
        sum_metrics["fnr"] = sum_metrics["fn"] / (sum_metrics["fn"] + sum_metrics["tp"])
        metrics["sum"] = sum_metrics
    except ZeroDivisionError as e:
        metrics["sum"] = {}
        print("ZERO DIVISION ERROR DURING SUM CALCULATION!")

    # this could be so easy, but unfortunately we have to deal with potentially undefined metrics!
    no_observations_per_metric = {}
    for fold, fold_metrics in metrics.items():
        if fold == "sum":
            continue
        for metric_name, metric_value in fold_metrics.items():
            if math.isnan(metric_value):
                no_observations_per_metric[metric_name] = no_observations_per_metric.get(metric_name, no_folds) - 1
                print(f"SKIPPED Metric {metric_name} during averaging Fold {fold}. Metric is NaN!")
                continue
            avg_metrics[metric_name] = avg_metrics.get(metric_name, 0) + metric_value

    avg_metrics = {
        metric_name: total_value / no_observations_per_metric.get(metric_name, no_folds)
        for metric_name, total_value in avg_metrics.items()
    }
    metrics["avg"] = avg_metrics

    with open(f"{results_path}/metrics_thresh_{threshold}.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    unsolved_instances = np.where(running_time_labels > feature_collection_cutoff)[0]
    timeout_labels_unsolved = timeout_labels[unsolved_instances]
    predictions_unsolved = predictions[unsolved_instances]

    create_confusion_matrix(predictions_unsolved, timeout_labels_unsolved, filename=filename_confusion_matrix)
    create_scatter_plot(predictions, running_time_labels, satisfiability_labels=verification_results,
                        y_label="Predicted Timeout", filename=filename_scatter_plot,
                        feature_collection_cutoff=feature_collection_cutoff)

    if running_times_timeout_prediction is None:
        assert feature_collection_cutoff
        running_times_timeout_prediction = running_time_labels.copy()
        running_times_timeout_prediction[predictions == 1.] = np.log10(feature_collection_cutoff)

    running_times_comparison = {
        "Vanilla Verifier": running_time_labels.tolist(),
        "Timeout Prediction": running_times_timeout_prediction.tolist()
    }
    results_timeout_prediction = verification_results.copy()
    results_timeout_prediction[predictions == 1.] = 2
    results_comparison = {
        "Vanilla Verifier": verification_results.tolist(),
        "Timeout Prediction": results_timeout_prediction.tolist()
    }

    create_ecdf_plot(
        running_times_all_verifiers=running_times_comparison,
        results_all_verifiers=results_comparison,
        filename=os.path.join(results_path, f"ecdf_threshold_{threshold}.png")
    )
