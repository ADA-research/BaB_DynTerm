# This file can be used to get results from an AB-Crown Log if the run has not completed successfully
import math
import re
from collections import defaultdict

import numpy as np

from src.util.io import load_log_file


def parse_oval_log(log_string):
    """
    Extracts running time and verification result of each instance given a verification log of Oval
    :param log_string: log as string
    :return: dict including running time and verification result for each instance
    """
    return_dict = {}
    split_by_instances = log_string.split("Verifying Image")
    for instance_lines in split_by_instances:
        index_number, time_spent, final_result = None, None, None
        for line in instance_lines.splitlines():
            if "############################" in line:
                pattern = r"(\d+)"

                match = re.search(pattern, line)

                if match:
                    index_number = int(match.group(1))
                    # print(f"Index: {index_number}")
            if "Final Result:" in line:

                pattern = r"Final Result: ([^,]+)\s.\sTime Taken: ([0-9.]+)"

                match = re.search(pattern, line)

                if match:
                    final_result = match.group(1)
                    time_spent = float(match.group(2))
                    # print("Final result:", final_result)
                    # print("Time spent:", time_spent)
            if "Skipped" in line:
                # print("Input Skipped!")
                final_result = "SKIPPED"
                # take very small number s.t. log10 is still defined
                time_spent = 0.00000000000000000000001

            if index_number is None or final_result is None or time_spent is None:
                continue
            else:
                return_dict[index_number] = {"result": final_result, "time": time_spent}

    return return_dict


def get_features_from_verification_log(log_string, bab_feature_cutoff=10, include_bab_features=True,
                                       total_neuron_count=None, frequency=None):
    """
    Extracts features of each instance given a Oval log string
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

    split_by_instances = log_string.split("Verifying Image")

    for instance_lines in split_by_instances:
        times_up = False
        batch_start_time = None
        last_checkpoint_passed = 0
        batch_count, prediction_margin, initial_min, initial_max, improved_min, improved_max, no_unstables, \
            cur_global_min, cur_global_max, visited_states, cur_no_domains, \
            tree_depth, positive_domain_percentage, time_taken_for_last_batch, time_since_last_batch = [0, np.inf] + [-np.inf] * 13
        index_number = -1
        lines = instance_lines.splitlines()
        for index, line in enumerate(lines):
            if "############################" in line:
                pattern = r"(\d+)"

                match = re.search(pattern, line)

                if match:
                    index_number = int(match.group(1))
                    # print(f"Index: {index_number}")
                    times_up = False

            # this has to happen before time cutoff because it may appear in log
            # after completed run (probably because of missing print flush)
            # However, it certainly is logged at the beginning of verification!
            if "Margin between highest and 2nd highest prediciton" in line:
                pattern = r'-?\d+\.\d+'
                match = re.search(pattern, line)

                if match:
                    prediction_margin = float(match.group())
                    # print("Prediction Margin", prediction_margin)

            if times_up:
                continue

            if "Elapsed Time" in line:
                pattern = r'-?\d+\.\d+'
                match = re.search(pattern, line)

                if match:
                    current_time = float(match.group())
                    if batch_start_time:
                        time_since_last_batch = current_time - batch_start_time
                    if frequency:
                        batch_count += 1
                        cur_features = [prediction_margin, initial_min, initial_max, improved_min, improved_max, no_unstables,
                                        cur_no_domains, visited_states, cur_global_min, cur_global_max, tree_depth, positive_domain_percentage, batch_count, time_since_last_batch, time_taken_for_last_batch]
                        if int(current_time) > last_checkpoint_passed + frequency:
                            last_checkpoint_passed = math.floor(current_time / frequency) * frequency
                        features[index_number][last_checkpoint_passed + frequency] = cur_features

                    elif int(current_time) > bab_feature_cutoff:
                        times_up = True


            if "New batch at" in line:
                pattern = r'-?\d+\.\d+'
                match = re.search(pattern, line)

                if match:
                    if batch_start_time:
                        time_taken_for_last_batch = float(match.group()) - batch_start_time
                    batch_start_time = float(match.group())
                    batch_count += 1

            if "Initial Bound (Linear Approx.)" in line:
                pattern = r'-?\d+\.\d*'

                match = re.findall(pattern, line)

                if match:
                    initial_min = float(match[0])
                    # print("GLOBAL MIN ", initial_min)
                    initial_max = float(match[1])
                    # print("GLOBAL MAX", initial_max)
            # Attention: Global LB found in both cases
            if "Global LB:" in line and "Initial Bound" not in line:
                pattern = r'-?\d+\.\d*'

                match = re.findall(pattern, line)

                if match:
                    improved_min = float(match[0])
                    # print("IMPROVED MIN ", improved_min)
                    improved_max = float(match[1])
                    # print("IMPROVED MAX", improved_max)

            if "Improvement margin for this problem and bounding algo" in line:
                pattern = r'[-+]?[0-9]*\.[0-9]*'
                match = re.search(pattern, line)

                if match:
                    improvement_margin = float(match.group(0))

            if "No. of Unstables:" in line:
                pattern = r'No. of Unstables:\s*(\d+)'

                match = re.search(pattern, line)

                if match:
                    no_unstables = int(match.group(1))
                    # print("No. Unstables", no_unstables)
            if "A batch of relu splits requires" in line:
                pattern = r'[-+]?[0-9]*\.[0-9]*'

                match = re.search(pattern, line)

                if match:
                    time_needed_for_relu_split = float(match.group(0))

            if "Branching requires" in line:
                pattern = r'[-+]?[0-9]*\.[0-9]*'
                match = re.search(pattern, line)

                if match:
                    time_needed_for_branching = float(match.group(0))

            # BaB Features
            if "Current: lb:" in line:
                pattern = r'[-+]?[0-9]*\.[0-9]*'
                match = re.findall(pattern, line)
                if match:
                    cur_global_min = float(match[0])
                    cur_global_max = float(match[1])
                    # print("BaB Cur Global Min", cur_global_min)
                    # print("BaB Cur Global Max", cur_global_max)

            if "Running Nb states visited:" in line:
                pattern = r'(\d+)'
                match = re.search(pattern, line)
                if match:
                    visited_states = int(match.group(1))
                    # print("no. of visited states", visited_states)

            if "Number of domains" in line:
                pattern = r'(\d+)'
                match = re.findall(pattern, line)
                if match:
                    cur_no_domains = int(match[0])
                    # print("cur no of domains", cur_no_domains)
                    cur_no_hard_domains = int(match[1])
                    # print("cur no of hard domains", cur_no_hard_domains)

            if "TREE DEPTH:" in line:
                pattern = r'(\d+)'
                match = re.findall(pattern, line)
                if match:
                    tree_depth = int(match[0])
                    # print("tree depth min", tree_depth_min)

            if "POSITIVE DOMAINS TOTAL PERCENTAGE" in line:
                pattern = r'[-+]?[0-9]*\.[0-9]*'
                match = re.search(pattern, line)

                if match:
                    positive_domain_percentage = float(match.group(0))

        if index_number >= 0:
            cur_features = [prediction_margin, initial_min, initial_max, improved_min, improved_max, no_unstables,
                                        cur_no_domains, visited_states, cur_global_min, cur_global_max, tree_depth, positive_domain_percentage, batch_count, time_since_last_batch, time_taken_for_last_batch]
            if frequency:
                features[index_number][last_checkpoint_passed + frequency] = cur_features
            else:
                features = features + cur_features
            # print(features)

    if frequency:
        max_checkpoints = int(max([max(instance.keys()) for instance in features.values()]) / frequency)
        combined_feature_dict = defaultdict(list)
        for index in features:
            last_features = [0, np.inf] + [-np.inf] * 17
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
    log_file = load_log_file("./verification_logs/MNIST_9_100/OVAL-BAB.log")
    features = get_features_from_verification_log(log_file, frequency=10, total_neuron_count=900)
    print(features)
