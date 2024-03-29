import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score

from src.util.visualization.diagrams import create_confusion_matrix, create_scatter_plot, create_ecdf_plot


def eval_fold(running_times, selection_running_times, results, selection_results, chosen_verifiers, verifiers,
              first_classification_at, ran_in_portfolio=None,
              ran_with_selected_algo=None, vbs_schedules_timeouts=False):
    """
    Evaluates a fold of algorithm selection results in comparison to the SBS and VBS.
    :param running_times: Running times of the single verifiers on the test instances
    :param selection_running_times: Running times taken by the algorithm selection approach per instance
    :param results: results of the respective verifiers on each instance
    :param selection_results: results of the algorithm selector on each instance
    :param chosen_verifiers: verifiers chosen by the algorithm selector
    :param verifiers: verifiers the experiment ran on
    :param first_classification_at: point in time to perform the first classification
    :param ran_in_portfolio: array of instance indices for which no algorithm selection was made, e.g. that ran in a parallel portfolio
    :param ran_with_selected_algo: array of instance indices for which an algorithm selection was made
    :param vbs_schedules_timeouts: how to calculate VBS: does it schedule timeouts or not?
    :return: dict containing running times/no. solved instances of SBS/VBS/Algo Select and accuracy scores of algo selection
    """

    no_verifiers = len(verifiers)
    verifier_running_times = [0 for _ in range(no_verifiers)]
    verifier_solved_instances = [0 for _ in range(no_verifiers)]
    test_samples = int(len(running_times) / no_verifiers)
    vbs_running_time = 0
    vbs_solved_instances = 0
    best_verifiers = []

    if ran_in_portfolio is None and ran_with_selected_algo is None:
        ran_in_portfolio = []
        ran_with_selected_algo = selection_results

    for test_index in range(test_samples):
        for verifier_index, verifier in enumerate(verifiers):
            verifier_running_times[verifier_index] += pow(10, running_times[test_index * no_verifiers + verifier_index])
            if results[test_index * no_verifiers + verifier_index] != 2:
                verifier_solved_instances[verifier_index] += 1
        vbs_index = np.argmin(running_times[test_index * no_verifiers:test_index * no_verifiers + no_verifiers])
        vbs_instance_running_time = np.min(
            running_times[test_index * no_verifiers:test_index * no_verifiers + no_verifiers])
        instance_results = results[test_index * no_verifiers:test_index * no_verifiers + no_verifiers]
        solved = 0. in instance_results or 1. in instance_results
        if solved:
            vbs_solved_instances += 1
            best_verifiers.append(vbs_index)
        else:
            best_verifiers.append(-1)

        if vbs_schedules_timeouts or solved:
            vbs_running_time += pow(10, vbs_instance_running_time)

    sbs_index = np.argmin(verifier_running_times)
    sbs = verifiers[sbs_index]
    sbs_running_time = verifier_running_times[sbs_index]
    sbs_solved_instances = verifier_solved_instances[sbs_index]

    print(f"SBS: {sbs} in {sbs_running_time} s ({sbs_running_time / 60} m)")
    print(f"SBS solved {sbs_solved_instances}!")

    print(f"VBS in {vbs_running_time} s ({vbs_running_time / 60} m)")
    print(f"VBS solved {vbs_solved_instances}!")

    # algorithm selection
    algo_select_running_time = sum(selection_running_times)
    algo_select_solved_instances = sum([1 if run_result in {0., 1.} else 0 for run_result in selection_results])
    print(f"Algorithm Selection Running Time: {algo_select_running_time} s ({algo_select_running_time / 60} m)")
    print(f"Algorithm Selection solved {algo_select_solved_instances}!")

    instances_ran_beyond_first_classification = [True for _ in best_verifiers]
    for test_index in range(len(best_verifiers)):
        if any([running_time < np.log10(first_classification_at) for running_time in
                running_times[test_index * no_verifiers:test_index * no_verifiers + no_verifiers]]):
            instances_ran_beyond_first_classification[test_index] = False
            ran_in_portfolio.remove(test_index)
    best_verifiers_beyond_first_classification = np.array(best_verifiers)[instances_ran_beyond_first_classification]
    percentage_ran_in_portfolio = len(ran_in_portfolio) / len(best_verifiers_beyond_first_classification)
    percentage_ran_with_selected_algo = len(ran_with_selected_algo) / len(best_verifiers_beyond_first_classification)
    best_verifiers_with_selected_algo = np.array(best_verifiers)[ran_with_selected_algo]
    chosen_verifiers_with_selected_algo = np.array(chosen_verifiers)[ran_with_selected_algo]
    algo_select_acc = accuracy_score(best_verifiers_with_selected_algo, chosen_verifiers_with_selected_algo)
    print(f"Accuracy of chosen Verifier in Algorithm Selection: {algo_select_acc}")

    all_timeout_indices = np.where(best_verifiers_with_selected_algo == -1)
    timeout_acc = accuracy_score(best_verifiers_with_selected_algo[all_timeout_indices],
                                 chosen_verifiers_with_selected_algo[all_timeout_indices])
    print(f"Accuracy in All Verifiers Timeout Predictions: {timeout_acc}")

    return {
        "sbs": sbs,
        "sbs_running_time": sbs_running_time,
        "sbs_solved": sbs_solved_instances,
        "vbs_running_time": vbs_running_time,
        "vbs_solved": vbs_solved_instances,
        "algo_select_running_time": algo_select_running_time,
        "algo_select_solved": algo_select_solved_instances,
        "algo_select_acc": algo_select_acc,
        "timeout_acc": timeout_acc,
        "percentage_ran_in_portfolio": percentage_ran_in_portfolio,
        "percentage_ran_with_selected_algo": percentage_ran_with_selected_algo
    }


def eval_final(fold_evals, fold_data, results_path, verifiers, threshold):
    """
    Function to aggregate fold metrics (sum and avg.) and to plot ECDF plot + Confusion Matrix
    :param fold_evals: dict containing all fold evaluations
    :param fold_data: Algorithm Selection and Verifier data for each fold (running times, verification results)
    :param results_path: where to store results
    :param verifiers: array of verifiers that ran the experiment
    :param threshold: confidence threshold a prediction had to exceed s.t. it was counted
    """
    sum_fold = defaultdict(int)
    for fold in fold_evals:
        for key, value in fold_evals[fold].items():
            if not isinstance(value, (int, float)):
                continue
            sum_fold[key] += value

    avg_fold = defaultdict(int)
    for metric in sum_fold:
        no_folds = len(list(fold_evals.keys()))
        avg_fold[metric] = sum_fold[metric] / no_folds

    fold_evals["avg"] = avg_fold
    fold_evals["sum"] = sum_fold

    with open(os.path.join(results_path, f"metrics_threshold_{threshold}.json"), 'w') as f:
        json.dump(fold_evals, f, indent=2)

    preds = np.array([])
    best_verifier_labels = np.array([])
    running_times_comparison = {
        verifier: [] for verifier in verifiers + ["Algorithm Selection"]
    }
    results_comparison = {
        verifier: [] for verifier in verifiers + ["Algorithm Selection"]
    }
    for fold in fold_data:
        preds = np.append(preds, fold_data[fold]["chosen_verifiers"])
        best_verifier_labels = np.append(best_verifier_labels, fold_data[fold]["best_verifiers"])
        for verifier_index, verifier in enumerate(verifiers):
            verifier_running_times = []
            verifier_results = []
            for i in range(len(fold_data[fold]["test_running_times"]) // len(verifiers)):
                verifier_running_times.append(
                    fold_data[fold]["test_running_times"][i * len(verifiers) + verifier_index])
                verifier_results.append(fold_data[fold]["results"][i * len(verifiers) + verifier_index])
            running_times_comparison[verifier] = running_times_comparison[verifier] + verifier_running_times
            results_comparison[verifier] = results_comparison[verifier] + verifier_results

        running_times_comparison["Algorithm Selection"] = np.append(running_times_comparison["Algorithm Selection"],
                                                                    np.log10(
                                                                        fold_data[fold]["selection_running_times"]))
        results_comparison["Algorithm Selection"] = results_comparison["Algorithm Selection"] + fold_data[fold][
            "selection_results"]

    filename_confusion_matrix = os.path.join(results_path, f"confusion_matrix_threshold_{threshold}.png")
    create_confusion_matrix(preds, best_verifier_labels, filename=filename_confusion_matrix)

    create_ecdf_plot(
        running_times_all_verifiers=running_times_comparison,
        results_all_verifiers=results_comparison,
        filename=os.path.join(results_path, f"ecdf_threshold_{threshold}.png")
    )
