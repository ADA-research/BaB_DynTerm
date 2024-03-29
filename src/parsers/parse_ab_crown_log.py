# This file can be used to get results from an AB-Crown Log if the run has not completed successfully
import math
import re
from collections import defaultdict

import numpy as np

from src.util.io import load_log_file


def parse_abcrown_log(log_string):
    """
    Extracts running time and verification result of each instance given a verification log of ab-CROWN
    :param log_string: log as string
    :return: dict including running time and verification result for each instance
    """
    return_dict = {}
    split_by_instances = log_string.split("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% idx:")
    for instance_lines in split_by_instances:
        index_number, seconds, result = None, None, None
        for line in instance_lines.splitlines():
            if "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" in line:
                pattern = r"vnnlib ID: (\d+)"

                match = re.search(pattern, line)

                if match:
                    index_number = int(match.group(1))
                    # print(f"Index: {index_number}")
            if "Result" in line:
                pattern = r"Result: ([\w-]+) in (\d+\.\d+) seconds"

                match = re.search(pattern, line)

                if match:
                    result = match.group(1)
                    seconds = float(match.group(2))
                    # print(f"Result: {result}, Time: {seconds} seconds")

        if index_number is None or result is None or seconds is None:
            continue
        else:
            return_dict[index_number] = {"result": result, "time": seconds}

    return return_dict


def get_features_from_verification_log(log_string, bab_feature_cutoff=10, include_bab_features=True, frequency=None,
                                       total_neuron_count=0):
    """
    Extracts features of each instance given a ab-CROWN log string
    :param log_string: string of ab-CROWN log
    :param bab_feature_cutoff: cutoff time for feature collection
    :param include_bab_features: if dynamic features (BaB-features) should be included
    :param frequency: if features should be collected at regular frequencies. Can be None, then bab_feature_cutoff is used as cutoff,
        else features are collected at regular checkpoints according to chosen frequency
    :param total_neuron_count: neuron count of network to be verified
    :return: if frequency is None: array of all features collected up to bab_feature_cutoff. If frequency is set,
        returns a dict with feature values for each checkpoint.
    """
    if frequency:
        features = defaultdict(dict)
    else:
        features = []
    split_by_instances = log_string.split("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% idx:")

    for instance_lines in split_by_instances:
        times_up = False
        last_checkpoint_passed = 0
        min_pgd_margin, crown_global_bound, alpha_crown_global_bound, no_unstables, \
            percentage_unstables, prediction_margin, positive_domain_ratio, domain_length, bab_lower_bound, \
            visited_domains, worst_bound_depth, bab_round, time_since_last_batch, \
            time_taken_for_last_batch = [-np.inf] * 14
        bab_start_time, cumulative_time = None, None
        index_number = -1
        lines = instance_lines.splitlines()
        for index, line in enumerate(lines):
            if "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" in line:
                pattern = r"vnnlib ID: (\d+)"

                match = re.search(pattern, line)

                if match:
                    index_number = int(match.group(1))
                    # print(f"Index: {index_number}")
                    times_up = False

            if times_up or index_number < 0:
                continue

            if "Elapsed Time:" in line:
                pattern = r'-?\d+\.\d+'
                match = re.search(pattern, line)

                if match:
                    current_time = float(match.group())
                    if cumulative_time and not bab_start_time:
                        bab_start_time = current_time - cumulative_time
                    if bab_start_time:
                        time_since_last_batch = current_time - (bab_start_time + cumulative_time)
                    if frequency:
                        if include_bab_features:
                            cur_features = [min_pgd_margin, crown_global_bound, alpha_crown_global_bound,
                                            no_unstables, percentage_unstables, prediction_margin,
                                            positive_domain_ratio, domain_length, visited_domains, bab_lower_bound,
                                            worst_bound_depth, bab_round, time_since_last_batch,
                                            time_taken_for_last_batch]
                        else:
                            cur_features = [min_pgd_margin, crown_global_bound, alpha_crown_global_bound,
                                            no_unstables, percentage_unstables, prediction_margin]
                        if int(current_time) > last_checkpoint_passed + frequency:
                            last_checkpoint_passed = math.floor(current_time / frequency) * frequency
                        features[index_number][last_checkpoint_passed + frequency] = cur_features

                    elif current_time > bab_feature_cutoff:
                        times_up = True

            if "PGD attack margin (first 2 examles and 10 specs):" in line:
                i = 1
                while 'device' not in line:
                    line = line + lines[index + i]
                    i += 1

                pattern = r'tensor\(\[\[\[(.*?)\]\]\]'

                match = re.search(pattern, line)

                if match:
                    matched_array = match.group(1)
                    pgd_attack_margin = eval(f"[{matched_array}]")
                    # print("MARGIN: ", pgd_attack_margin)
                    min_pgd_margin = min(pgd_attack_margin)
            if "initial CROWN bounds" in line:
                i = 1
                while 'device' not in line:
                    line = line + lines[index + i]
                    i += 1

                pattern = r'tensor\(\[\s*(.*?)\s*\],\s*device'

                match = re.search(pattern, line)

                if match:
                    matched_array = match.group(1)
                    initial_crown_bounds = eval(matched_array)
                    # print("CROWN ", initial_crown_bounds)
                    crown_global_bound = min(initial_crown_bounds)

            if "initial alpha-crown bounds:" in line:
                i = 1
                while 'device' not in line:
                    line = line + lines[index + i]
                    i += 1
                pattern = r'tensor\(\[\s*(.*?)\s*\],\s*device'

                match = re.search(pattern, line)

                if match:
                    matched_array = match.group(1)
                    if "inf" in matched_array:
                        matched_array = matched_array.replace("inf", "np.inf")
                    initial_alpha_crown_bounds = eval(matched_array)
                    alpha_crown_global_bound = min(initial_alpha_crown_bounds)

            if "# of unstable neurons" in line:
                line = line + lines[index + 1]
                pattern = r'# of unstable neurons:\s*(\d+)'

                match = re.search(pattern, line)

                if match:
                    no_unstables = int(match.group(1))
                    # print("No. Unstables", no_unstables)
                    percentage_unstables = no_unstables / total_neuron_count
                    # print("Percentage Unstables", percentage_unstables)
            if "Model prediction is:" in line:
                line = line + lines[index + 1] + lines[index + 2]
                pattern = r'tensor\(\[\s*(.*?)\s*\],\s*device'
                match = re.search(pattern, line)

                if match:
                    model_prediction = list(eval(match.group(1)))
                    # print("Model Prediction", model_prediction)
                    model_prediction.sort(reverse=True)
                    prediction_margin = model_prediction[0] - model_prediction[1]
                    # print("Prediciton Margin", prediction_margin)

            # BaB Features
            if "ratio of positive domain" in line:
                pattern = r'=\s*([\d.]+)$'
                match = re.search(pattern, line)
                if match:
                    positive_domain_ratio = float(match.group(1))
                    # print("RATIO", positive_domain_ratio)

            if "length of domains" in line:
                pattern = r'length of domains:\s*(\d+)'
                match = re.search(pattern, line)

                if match:
                    domain_length = int(match.group(1))
                    # print("DOMAIN LENGTH", domain_length)

            if "domains visited" in line:
                pattern = r'(\d+) domains visited'
                match = re.search(pattern, line)

                if match:
                    visited_domains = int(match.group(1))
                    # print("DOMAIN LENGTH", domain_length)

            if "Current (lb-rhs)" in line:
                pattern = r'-?\d+\.\d+'
                match = re.search(pattern, line)

                if match:
                    bab_lower_bound = float(match.group())
                    # print("BAB LOWER BOUND", bab_lower_bound)

            if "Current worst splitting domains" in line:
                domain_line = lines[index + 1]
                worst_domain = domain_line.split(",")[0]
                match = re.search(r"\((\d+)\)", worst_domain)

                if match:
                    worst_bound_depth = int(match.group(1))
                    # print("BAB LOWER BOUND", bab_lower_bound)
            if "BaB round" in line:
                match = re.search(r"(\d+)", line)

                if match:
                    bab_round = int(match.group(1))

            if "Cumulative time:" in line:
                match = re.search(r"Cumulative time: ([\d+\.\d]+)", line)

                if match:
                    if cumulative_time:
                        time_taken_for_last_batch = float(match.group(1)) - cumulative_time
                    cumulative_time = float(match.group(1))

        if index_number >= 0:
            if include_bab_features:
                cur_features = [min_pgd_margin, crown_global_bound, alpha_crown_global_bound, no_unstables,
                                percentage_unstables, prediction_margin, positive_domain_ratio, domain_length,
                                visited_domains, bab_lower_bound, worst_bound_depth, bab_round, time_since_last_batch,
                                time_taken_for_last_batch]
            else:
                cur_features = [min_pgd_margin, crown_global_bound, alpha_crown_global_bound, no_unstables,
                                percentage_unstables, prediction_margin]
            if frequency:
                features[index_number][last_checkpoint_passed + frequency] = cur_features
            else:
                features = features + cur_features

    if frequency:
        max_checkpoints = int(max([max(instance.keys()) for instance in features.values()]) / frequency)
        combined_feature_dict = defaultdict(list)
        for index in features:
            last_features = [-np.inf] * 14
            for checkpoint in range(frequency, (max_checkpoints + 1) * frequency, frequency):
                checkpoint_features = features[index].get(checkpoint)
                if not checkpoint_features:
                    combined_feature_dict[checkpoint] = combined_feature_dict[checkpoint] + last_features
                else:
                    last_features = checkpoint_features
                    combined_feature_dict[checkpoint] = combined_feature_dict[checkpoint] + checkpoint_features
        for checkpoint, checkpoint_features in combined_feature_dict.items():
            combined_feature_dict[checkpoint] = np.reshape(checkpoint_features, (index_number + 1, -1))
        features = combined_feature_dict
    else:
        features = np.reshape(features, (index_number + 1, -1))
    return features

