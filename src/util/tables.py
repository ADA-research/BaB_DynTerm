import json
import os

import numpy as np

from src.util.constants import experiment_groups, TIMEOUT, UNSAT, SUPPORTED_VERIFIERS, experiment_samples


def create_running_time_regression_table(results_path):
    """
    Creates table in CSV format for running time regression experiments.
    :param results_path: path with results of experiment
    :return: table in CSV format
    """
    csv = r"Verifiers,$\alpha\beta$-CROWN,,,VeriNet,,,Oval,," + "\n"
    csv += r"Metrics,RMSE,R2,$\rho$,RMSE,R2,$\rho$,RMSE,R2,$\rho$" + "\n"
    for experiment_group, experiments in experiment_groups.items():
        for experiment in experiments:
            csv += f"{experiment},"
            for verifier in ["ABCROWN", "VERINET", "OVAL-BAB"]:
                experiment_path = f"./{results_path}/{experiment}/{verifier}"
                if not os.path.exists(f"{experiment_path}/running_time_prediction_metrics.json"):
                    csv += f'-,-,-,'
                    continue
                with open(f"{experiment_path}/running_time_prediction_metrics.json", 'r') as f:
                    verifier_data = json.load(f)
                avg_fold = verifier_data["avg"]
                for key, value in avg_fold.items():
                    if isinstance(value, float):
                        avg_fold[key] = round(value, 2)
                csv += f'{avg_fold["rmse_test_right"]}, {avg_fold["r2_test_right"]}, {avg_fold["spearman_correlation_test_right"]},'
            csv += "\n"
    return csv


def create_timeouts_table(results_path, thresholds):
    """
    Creates table in CSV format for timeout prediction experiment.
    :param results_path: path with results of experiment
    :param thresholds: thresholds that should be included in table.
    :return: table in CSV format
    """
    csv = r"$\theta$,,"
    for threshold in thresholds:
        csv += f"{threshold},,,,,,,,,"
    csv += "\n"
    csv += r",,"
    for _ in thresholds:
        csv += r"$\alpha\beta$-CROWN,,,VeriNet,,,Oval,,,"
    csv += "\n"

    csv += "Benchmark Group, Benchmark Name, Acc.,TPR,FPR, Acc.,TPR,FPR,Acc.,TPR,FPR,Acc.,TPR,FPR,Acc.,TPR,FPR,Acc.,TPR,FPR\n"

    for experiment_group, experiments in experiment_groups.items():
        for experiment in experiments:
            csv += f'{experiment_group}, {experiment},'
            for thresh in thresholds:
                for verifier in ["ABCROWN", "VERINET", "OVAL-BAB"]:
                    experiment_path = f"./{results_path}/{experiment}/{verifier}"
                    if not os.path.exists(f"{experiment_path}/metrics_thresh_{thresh}.json"):
                        csv += f'-,-,-,'
                        continue
                    with open(f"{experiment_path}/metrics_thresh_{thresh}.json", 'r') as f:
                        verifier_data = json.load(f)
                    avg_fold = verifier_data["avg"]
                    for key, value in avg_fold.items():
                        if isinstance(value, float):
                            avg_fold[key] = round(value, 2)
                    csv += f'{avg_fold["test_acc"]}, {avg_fold["tpr"]}, {avg_fold["fpr"]}, '
            csv += '\n'

    return csv
def create_benchmark_overview_table(results_path):
    """
    Creates table in CSV format for premature termination of presumed timeouts.
    :param results_path: path with results of experiment
    :param thresholds: thresholds that should be included in table.
    :return: table in CSV format
    """
    thresholds = [0.99]
    csv = ',,'
    for threshold in thresholds:
        csv += rf"$\theta$={threshold},,,,,,,,,,,,,,"
    csv += '\n'
    csv += ',,'
    for _ in thresholds:
        csv += r"$\alpha\beta$-CROWN,,,,,VeriNet,,,,,Oval,,,,," + "\n"
    csv += "Benchmark,,Running Time [GPU h],,# Solved,,,Running Time [GPU h],,# Solved,,,Running Time [GPU h],,# Solved\n"
    for experiment_group, experiments in experiment_groups.items():
        for experiment in experiments:
            for thresh in thresholds:
                experiment_path = f"{results_path}/{experiment}"
                no_instances = experiment_samples[experiment]
                csv += f'{experiment},,'
                for verifier in SUPPORTED_VERIFIERS:
                    verifier_results_path = f"./{experiment_path}/{verifier}"
                    if not os.path.exists(f"{verifier_results_path}/ecdf_threshold_{thresh}.png.json"):
                        csv += f'-,,-,,,'
                        continue
                    with open(f"{verifier_results_path}/ecdf_threshold_{thresh}.png.json", 'r') as f:
                        verifier_data = json.load(f)
                    vanilla_running_times = verifier_data["running_times"]["Vanilla Verifier"]
                    vanilla_results = verifier_data["results"]["Vanilla Verifier"]
                    # one of the hustles one has to do because abCROWN did not filter misclassified instances
                    if no_instances != len(vanilla_results):
                        misclassified = len(vanilla_results) - no_instances
                        no_timeouts_vanilla = sum([1 for result in vanilla_results if result == TIMEOUT])
                        no_solved_vanilla = sum([1 for result in vanilla_results if result != TIMEOUT]) - misclassified
                        no_verified_vanilla = sum([1 for result in vanilla_results if result == UNSAT])
                        timeout_termination_running_times = verifier_data["running_times"]["Timeout Prediction"]
                        timeout_termination_results = verifier_data["results"]["Timeout Prediction"]
                        no_timeouts_timeout_termination = sum(
                            [1 for result in timeout_termination_results if result == 2.])
                        no_solved_timeout_termination = sum(
                            [1 for result in timeout_termination_results if result != 2.]) - misclassified
                    else:
                        no_timeouts_vanilla = sum([1 for result in vanilla_results if result == TIMEOUT])
                        no_solved_vanilla = sum([1 for result in vanilla_results if result != TIMEOUT])
                        no_verified_vanilla = sum([1 for result in vanilla_results if result == UNSAT])
                        timeout_termination_running_times = verifier_data["running_times"]["Timeout Prediction"]
                        timeout_termination_results = verifier_data["results"]["Timeout Prediction"]
                        no_timeouts_timeout_termination = sum(
                            [1 for result in timeout_termination_results if result == 2.])
                        no_solved_timeout_termination = sum(
                            [1 for result in timeout_termination_results if result != 2.])

                    wct_vanilla = sum([pow(10, log_running_time) for log_running_time in vanilla_running_times])
                    wct_timeout_termination = sum(
                        [pow(10, log_running_time) for log_running_time in timeout_termination_running_times])

                    improv_percentage = (wct_timeout_termination / wct_vanilla) * 100

                    solved_difference = no_solved_timeout_termination - no_solved_vanilla

                    if solved_difference < 0:
                        diff_sign = ""
                    elif solved_difference > 0:
                        diff_sign = "$+$"
                    else:
                        diff_sign = r"$\pm$"

                    csv += f'{round(wct_vanilla / 60 / 60, 2)},,{no_solved_vanilla},,'

            csv += '\n'

    return csv

def create_timeout_termination_table(results_path, thresholds):
    """
    Creates table in CSV format for premature termination of presumed timeouts.
    :param results_path: path with results of experiment
    :param thresholds: thresholds that should be included in table.
    :return: table in CSV format
    """
    csv = ',,'
    for threshold in thresholds:
        csv += rf"$\theta$={threshold},,,,,,,,,,,,,,"
    csv += '\n'
    csv += ',,'
    for _ in thresholds:
        csv += r"$\alpha\beta$-CROWN,,,,,VeriNet,,,,,Oval,,,,," + "\n"
    csv += "Benchmark,,Running Time [GPU h],,# Solved,,,Running Time [GPU h],,# Solved,,,Running Time [GPU h],,# Solved\n"
    for experiment_group, experiments in experiment_groups.items():
        for experiment in experiments:
            for thresh in thresholds:
                experiment_path = f"{results_path}/{experiment}"
                no_instances = experiment_samples[experiment]
                csv += f'{experiment},,'
                for verifier in SUPPORTED_VERIFIERS:
                    verifier_results_path = f"./{experiment_path}/{verifier}"
                    if not os.path.exists(f"{verifier_results_path}/ecdf_threshold_{thresh}.png.json"):
                        csv += f'-,,-,,,'
                        continue
                    with open(f"{verifier_results_path}/ecdf_threshold_{thresh}.png.json", 'r') as f:
                        verifier_data = json.load(f)
                    vanilla_running_times = verifier_data["running_times"]["Vanilla Verifier"]
                    vanilla_results = verifier_data["results"]["Vanilla Verifier"]
                    # one of the hustles one has to do because abCROWN did not filter misclassified instances
                    if no_instances != len(vanilla_results):
                        misclassified = len(vanilla_results) - no_instances
                        no_timeouts_vanilla = sum([1 for result in vanilla_results if result == TIMEOUT])
                        no_solved_vanilla = sum([1 for result in vanilla_results if result != TIMEOUT]) - misclassified
                        no_verified_vanilla = sum([1 for result in vanilla_results if result == UNSAT])
                        timeout_termination_running_times = verifier_data["running_times"]["Timeout Prediction"]
                        timeout_termination_results = verifier_data["results"]["Timeout Prediction"]
                        no_timeouts_timeout_termination = sum(
                            [1 for result in timeout_termination_results if result == 2.])
                        no_solved_timeout_termination = sum(
                            [1 for result in timeout_termination_results if result != 2.]) - misclassified
                    else:
                        no_timeouts_vanilla = sum([1 for result in vanilla_results if result == TIMEOUT])
                        no_solved_vanilla = sum([1 for result in vanilla_results if result != TIMEOUT])
                        no_verified_vanilla = sum([1 for result in vanilla_results if result == UNSAT])
                        timeout_termination_running_times = verifier_data["running_times"]["Timeout Prediction"]
                        timeout_termination_results = verifier_data["results"]["Timeout Prediction"]
                        no_timeouts_timeout_termination = sum(
                            [1 for result in timeout_termination_results if result == 2.])
                        no_solved_timeout_termination = sum(
                            [1 for result in timeout_termination_results if result != 2.])

                    wct_vanilla = sum([pow(10, log_running_time) for log_running_time in vanilla_running_times])
                    wct_timeout_termination = sum(
                        [pow(10, log_running_time) for log_running_time in timeout_termination_running_times])

                    improv_percentage = (wct_timeout_termination / wct_vanilla) * 100

                    solved_difference = no_solved_timeout_termination - no_solved_vanilla

                    if solved_difference < 0:
                        diff_sign = "$-$"
                    elif solved_difference > 0:
                        diff_sign = "$+$"
                    else:
                        diff_sign = r"$\pm$"

                    csv += f'{round(wct_timeout_termination / 60 / 60, 2)},({round(improv_percentage)}%),{no_solved_timeout_termination},({diff_sign}{abs(solved_difference)}),,'

            csv += '\n'

    return csv


def create_algorithm_selection_table(results_path, thresholds):
    """
    Creates table in CSV format for algorithm selection results.
    :param results_path: path with results of experiment
    :param thresholds: thresholds that should be included in table.
    :return: table in CSV format
    """
    csv = r"$\theta$,,0.5,,,,,,0.99,,,,,," + "\n"
    csv += "Benchmark,, Running Time [GPU h],,# Solved,,Portfolio [%],,Running Time [GPU h],,# Solved,,Portfolio [%],\n"
    improvements = []
    for experiment_group, experiments in experiment_groups.items():
        for experiment in experiments:
            if experiment == "VIT":
                continue
            csv += f'{experiment},,'
            no_instances = experiment_samples[experiment]
            for thresh in thresholds:
                experiment_path = f"./{results_path}/{experiment}"
                if not os.path.exists(f"{experiment_path}/ecdf_threshold_{thresh}.png.json"):
                    csv += f'-,-,'
                    continue
                with open(f"{experiment_path}/ecdf_threshold_{thresh}.png.json", 'r') as f:
                    verifier_data = json.load(f)
                # TODO: This should not be hardcoded but find the SBS!
                vanilla_running_times = verifier_data["running_times"]["ABCROWN"]
                vanilla_results = verifier_data["results"]["ABCROWN"]
                misclassified = len(vanilla_results) - no_instances
                no_timeouts_vanilla = sum([1 for result in vanilla_results if result == TIMEOUT])
                no_solved_vanilla = sum([1 for result in vanilla_results if result != TIMEOUT]) - misclassified
                no_verified_vanilla = sum([1 for result in vanilla_results if result == UNSAT])
                timeout_termination_running_times = verifier_data["running_times"]["Algorithm Selection"]
                timeout_termination_results = verifier_data["results"]["Algorithm Selection"]
                no_timeouts_timeout_termination = sum(
                    [1 for result in timeout_termination_results if result == 2.])
                no_solved_timeout_termination = sum(
                    [1 for result in timeout_termination_results if result != 2.]) - misclassified

                wct_vanilla = sum([pow(10, log_running_time) for log_running_time in vanilla_running_times])
                wct_timeout_termination = sum(
                    [pow(10, log_running_time) for log_running_time in timeout_termination_running_times])

                improv_percentage = (wct_timeout_termination / wct_vanilla) * 100

                if thresh == .99:
                    improvements = improvements + [improv_percentage]

                solved_difference = no_solved_timeout_termination - no_solved_vanilla

                if solved_difference < 0:
                    diff_sign = ""
                elif solved_difference > 0:
                    diff_sign = "+"
                else:
                    diff_sign = "+-"

                with open(f"{experiment_path}/metrics_threshold_{thresh}.json", 'r') as f:
                    metrics = json.load(f)

                avg_metrics = metrics["avg"]
                percentage_ran_in_portfolio = avg_metrics["percentage_ran_in_portfolio"]
                percentage_ran_with_selected_algo = avg_metrics["percentage_ran_with_selected_algo"]

                csv += f'{round(wct_timeout_termination / 60 / 60, 2)},({round(improv_percentage)}%), {no_solved_timeout_termination},({diff_sign}{solved_difference}), {round(percentage_ran_in_portfolio * 100, 2)},,'

            csv += '\n'

    #print(improvements)
    #print(f"Average Improvement:", sum(improvements) / len(improvements))

    return csv