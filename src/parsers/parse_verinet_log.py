# This file can be used to get results from an AB-Crown Log if the run has not completed successfully
import math
import re
from collections import defaultdict

import numpy as np

from src.util.io import load_log_file


def parse_verinet_log(log_string):
    """
    Extracts running time and verification result of each instance given a verification log of VeriNet
    :param log_string: log as string
    :return: dict including running time and verification result for each instance
    """
    return_dict = {}
    split_by_instances = log_string.split("Verifying image")
    for instance_lines in split_by_instances:
        index_number, time_spent, final_result = None, None, None
        for line in instance_lines.splitlines():
            if "#######################################################" in line:
                pattern = r"(\d+) #######################################################"

                match = re.search(pattern, line)

                if match:
                    index_number = int(match.group(1))
                    # print(f"Index: {index_number}")
            if "Final result of input" in line:
                pattern_result = r"Final result of input \d+: ([^,]+)"
                pattern_time = r"time spent: ([0-9.]+) seconds"
                result_match = re.search(pattern_result, line)
                time_match = re.search(pattern_time, line)
                # Check if a match was found
                if result_match:
                    final_result = result_match.group(1)
                    if final_result == "Skipped":
                        final_result = "Status.Skipped"
                        time_spent = 0.00000000000000000000001
                    # print("Final result:", final_result)
                    # print("Time spent:", time_spent)
                if time_match:
                    time_spent = float(time_match.group(1))

            if "Network" in line and "Property" in line and "Status" in line and "time" in line:
                pattern_result = r"Status:\s([^,]+), branches"
                pattern_time = r"time: ([0-9.]+) seconds"
                result_match = re.search(pattern_result, line)
                time_match = re.search(pattern_time, line)
                if result_match:
                    final_result = result_match.group(1)
                    if final_result == "Skipped":
                        final_result = "Status.Skipped"
                        time_spent = 0.00000000000000000000001
                if time_match:
                    time_spent = float(time_match.group(1))
                if not result_match and not time_match:
                    final_result = "Status.Skipped"
                    # take very small number s.t. logarithm is still defined
                    time_spent = 0.00000000000000000000001

            if index_number is None or final_result is None or time_spent is None:
                continue
            else:
                return_dict[index_number] = {"result": final_result, "time": time_spent}

    return return_dict


def get_features_from_verification_log(log_string, bab_feature_cutoff=10, include_bab_features=True,
                                       total_neuron_count=None, frequency=None, no_classes=10):
    """
    Extracts features of each instance given a VeriNet log string
    :param log_string: string of ab-CROWN log
    :param bab_feature_cutoff: cutoff time for feature collection
    :param include_bab_features: if dynamic features (BaB-features) should be included
    :param frequency: if features should be collected at regular frequencies. Can be None, then bab_feature_cutoff is used as cutoff,
        else features are collected at regular checkpoints according to chosen frequency
    :param total_neuron_count: neuron count of network to be verified
    :param no_classes: Number of classes that given network has. Needed to calculate fraction of safe constraints.
    :return: if frequency is None: array of all features collected up to bab_feature_cutoff. If frequency is set,
        returns a dict with feature values for each checkpoint.
    """

    if frequency:
        features = defaultdict(dict)
    else:
        features = []

    split_by_instances = log_string.split("#################################################### Verifying image")

    for instance_lines in split_by_instances:
        times_up = False
        last_checkpoint_passed = 0
        min_attack_margin, one_shot_global_bound_min, one_shot_global_max, no_unstables, percentage_unstables, \
            prediction_margin, positive_domain_ratio, total_branches, explored_branches, \
            bab_cur_lower_bound, tree_depth, one_shot_safe_percentage, time_since_last_report = [np.inf] + [
            -np.inf] * 12
        index_number = -1
        lines = instance_lines.splitlines()
        current_time = 0
        for index, line in enumerate(lines):
            if "#######################################################" in line:
                pattern = r"(\d+) #######################################################"

                match = re.search(pattern, line)

                if match:
                    index_number = int(match.group(1))
                    # print(f"Index: {index_number}")
                    times_up = False

            # this has to happen before time cutoff because it may happen after BaB (probably because of missing print flush)
            if "Margin between highest and 2nd highest prediciton" in line:
                pattern = r'-?\d+\.\d+'
                match = re.search(pattern, line)

                if match:
                    prediction_margin = float(match.group())
                    # print("Prediction Margin", prediction_margin)

            if times_up:
                continue

            if "ELAPSED TIME:" in line:
                pattern = r'-?\d+\.\d+'
                match = re.search(pattern, line)

                if match:
                    # Hack to prevent taking last timing of last instance as new timing
                    if abs(current_time - float(match.group())) > 60:
                        continue
                    current_time = float(match.group())
                    if frequency:
                        time_since_last_report = math.ceil(current_time / frequency) * frequency - current_time
                    elif not (bab_feature_cutoff < current_time < bab_feature_cutoff + 10):
                        time_since_last_report = bab_feature_cutoff - current_time

                    # Hack that is needed because in log the last BaB Log can occur after Instance has changed to new one
                    if frequency:
                        if int(current_time) > last_checkpoint_passed + frequency:
                            last_checkpoint_passed = math.floor(current_time / frequency) * frequency
                        cur_features = [min_attack_margin,
                                        one_shot_global_bound_min, one_shot_global_max, one_shot_safe_percentage,
                                        no_unstables, percentage_unstables, prediction_margin, positive_domain_ratio,
                                        total_branches, explored_branches, bab_cur_lower_bound, tree_depth,
                                        time_since_last_report]
                        features[index_number][last_checkpoint_passed + frequency] = cur_features

                    elif bab_feature_cutoff < current_time < bab_feature_cutoff + 10:
                        times_up = True

            if "Max Attack Margin:" in line:
                pattern = r'[-+]?\d+\.\d+'

                match = re.search(pattern, line)

                if match:
                    # Here we negate margin for compatibility with abCROWN ,
                    # as my calculation is pred - pred[correct] (and not other way around as in abCROWN)
                    cur_min_attack_margin = float(match.group()) * -1
                    min_attack_margin = min(min_attack_margin, cur_min_attack_margin)

            if "One-Shot Safe Constraints" in line:

                pattern = r'\[([\d,\s]+)\]'

                match = re.search(pattern, line)

                if match:
                    array_str = match.group(1)

                    one_shot_safe_constraints = [int(num) for num in re.split(r',\s*', array_str)]

                    # print("One-Shot Safe constraints:", one_shot_safe_constraints)
                    one_shot_safe_percentage = len(one_shot_safe_constraints) / (no_classes - 1)
                    # print("Safe Percentage:", one_shot_safe_percentage)

            if "Impact Score" in line:
                exponent_pattern = r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?"
                digit_pattern = r'(\d+\.\d+)'
                exponent_matches = re.findall(exponent_pattern, line)
                digit_matches = re.findall(digit_pattern, line)

                if exponent_matches and digit_matches:
                    values = [float(str(digit) + str(exponent)) for digit, exponent in
                              zip(digit_matches, exponent_matches)]
                    impact_max = values[0]
                    # print("Max:", values[0])
                    impact_min = values[1]
                    # print("Min:", values[1])
                    impact_med = values[2]
                    # print("Median:", values[2])
                    impact_mean = values[3]
                    # print("Mean:", values[3])

            if "GLOBAL MIN" in line:
                pattern = r'-?\d+\.\d+'

                match = re.search(pattern, line)

                if match:
                    global_min = float(match.group())
                    # print("GLOBAL MIN ", global_min)
                    one_shot_global_bound_min = global_min
            if "GLOBAL MAX:" in line:
                pattern = r'-?\d+\.\d+'

                match = re.search(pattern, line)

                if match:
                    global_max = float(match.group())
                    # print("GLOBAL MAX ", global_max)
                    one_shot_global_max = global_max

            if "No. Unstables:" in line:
                pattern = r'No. Unstables:\s*(\d+)'

                match = re.search(pattern, line)

                if match:
                    no_unstables = int(match.group(1))
                    # print("No. Unstables", no_unstables)
                    percentage_unstables = no_unstables / total_neuron_count
                    # print("Percentage Unstables", percentage_unstables)

            # BaB Features
            if "FRACTION OF POSITIVE DOMAIN:" in line:
                pattern = r'-?\d+\.\d+'
                match = re.search(pattern, line)
                if match:
                    positive_domain_ratio = float(match.group())
                    # print("RATIO", positive_domain_ratio)

            if "Total Branches:" in line:
                pattern = r'(\d+)'
                match = re.search(pattern, line)
                if match:
                    total_branches = int(match.group(1))
                    # print("total branches", total_branches)

            if "Explored Branches:" in line:
                pattern = r'(\d+)'
                match = re.search(pattern, line)
                if match:
                    explored_branches = int(match.group(1))
                    # print("explored branches", explored_branches)

            if "Depth of Tree:" in line:
                pattern = r'(\d+)'
                match = re.search(pattern, line)
                if match:
                    tree_depth = int(match.group(1))
                    # print("tree depth", tree_depth)

            if "CURRENT GLOBAL LOWER BOUND:" in line:
                pattern = r'-?\d+\.\d+'
                match = re.search(pattern, line)
                if match:
                    bab_cur_lower_bound = float(match.group())
                    # print("BAB CUR LOWER BOUND", bab_cur_lower_bound)

        if index_number >= 0:
            cur_features = [min_attack_margin, one_shot_global_bound_min, one_shot_global_max, one_shot_safe_percentage,
                            no_unstables, percentage_unstables, prediction_margin, positive_domain_ratio,
                            total_branches, explored_branches, bab_cur_lower_bound, tree_depth, time_since_last_report]
            if frequency:
                features[index_number][last_checkpoint_passed + frequency] = cur_features
            else:
                features = features + cur_features

    if frequency:
        max_checkpoints = int(max([max(instance.keys()) for instance in features.values()]) / frequency)
        combined_feature_dict = defaultdict(list)
        for index in features:
            last_features = [np.inf] + [-np.inf] * 12
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


if __name__ == "__main__":
    log_file = load_log_file("./verification_logs/OVAL21/VERINET.log")
    features = get_features_from_verification_log(log_file, frequency=None, total_neuron_count=600,
                                                  bab_feature_cutoff=35)
    print(features)
