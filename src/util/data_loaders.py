from collections import defaultdict

import numpy as np

from src.parsers.parse_ab_crown_log import parse_abcrown_log
from src.util.io import load_log_file
from src.parsers.parse_ab_crown_log import \
    get_features_from_verification_log as get_features_from_verification_log_abcrown
from src.parsers.parse_verinet_log import \
    get_features_from_verification_log as get_features_from_verification_log_verinet, parse_verinet_log
from src.parsers.parse_oval_log import \
    get_features_from_verification_log as get_features_from_verification_log_oval, parse_oval_log
from src.util.constants import result_to_enum, ABCROWN, VERINET, OVAL


def load_abcrown_data(log_path, artificial_cutoff=None, par=1, features_from_log=True, feature_path=None,
                      neuron_count=0, no_classes=10,
                      feature_collection_cutoff=10, filter_misclassified=False, frequency=None):
    """
    Function to return features, running times and verification results based on a log file of ab-CROWN
    :param log_path: path to log file
    :param artificial_cutoff: cutoff time for verification, may be lower than actual cutoff time
    :param par: PAR score, i.e. factor by which the running times of unsolved instances are penalised
    :param features_from_log: if features should be parsed from the log file or if they should be loaded from a numpy array
    :param feature_path: must be provided if features_from_log is False. Path to numpy array of features.
    :param neuron_count: neuron count of verified network
    :param no_classes: number of output classes of verified network
    :param feature_collection_cutoff: seconds for which features should be collected
    :param filter_misclassified: if misclassified instances should be included or not
    :param frequency: frequencies for which features should be provided
    :return: features, running times, verification results in clear text and verification results as enum.
        If frequency is not None, features are a dict where features are provided for each checkpoint, else a numpy array.
    """
    running_time_dict = parse_abcrown_log(load_log_file(log_path))
    if features_from_log:
        features = get_features_from_verification_log_abcrown(load_log_file(log_path),
                                                              bab_feature_cutoff=feature_collection_cutoff,
                                                              frequency=frequency, total_neuron_count=neuron_count)
    else:
        features = np.load(feature_path, allow_pickle=True)
    running_times = [value["time"] for key, value in running_time_dict.items()]
    results = [value["result"] for key, value in running_time_dict.items()]

    for i in range(len(running_times)):
        if artificial_cutoff and running_times[i] > artificial_cutoff:
            running_times[i] = artificial_cutoff
            results[i] = "unknown"
        if result_to_enum[results[i]] == 2:
            running_times[i] = running_times[i] * par

    enum_results = [result_to_enum[result] for result in results]

    # log10 running times for easier approximation
    running_times = np.log10(running_times)

    # Replace -np.inf or np.inf values (non-existent features) with 0
    if frequency:
        for checkpoint, checkpoint_features in features.items():
            checkpoint_features[checkpoint_features == -np.inf] = 0
            checkpoint_features[checkpoint_features == np.inf] = 0
    else:
        features[features == -np.inf] = 0
        features[features == np.inf] = 0

    return features, running_times, results, np.array(enum_results)


def load_verinet_data(log_path, artificial_cutoff=None, par=1, feature_path=None,
                      filter_misclassified=False, neuron_count=0, feature_collection_cutoff=10, frequency=None,
                      no_classes=10, features_from_log=True):
    """
    Function to return features, running times and verification results based on a log file of VeriNet
    :param log_path: path to log file
    :param artificial_cutoff: cutoff time for verification, may be lower than actual cutoff time
    :param par: PAR score, i.e. factor by which the running times of unsolved instances are penalised
    :param features_from_log: if features should be parsed from the log file or if they should be loaded from a numpy array
    :param feature_path: must be provided if features_from_log is False. Path to numpy array of features.
    :param neuron_count: neuron count of verified network
    :param no_classes: number of output classes of verified network
    :param feature_collection_cutoff: seconds for which features should be collected
    :param filter_misclassified: if misclassified instances should be included or not
    :param frequency: frequencies for which features should be provided
    :return: features, running times, verification results in clear text and verification results as enum.
        If frequency is not None, features are a dict where features are provided for each checkpoint, else a numpy array.
    """
    running_time_dict = parse_verinet_log(load_log_file(log_path))
    log_string = load_log_file(log_path)
    features = get_features_from_verification_log_verinet(log_string, total_neuron_count=neuron_count,
                                                          bab_feature_cutoff=feature_collection_cutoff,
                                                          frequency=frequency, no_classes=no_classes)
    running_times = [value["time"] for key, value in running_time_dict.items()]
    results = [value["result"] for key, value in running_time_dict.items()]

    for i in range(len(running_times)):
        if artificial_cutoff and running_times[i] > artificial_cutoff:
            running_times[i] = artificial_cutoff
            results[i] = "Status.Undecided"

        if result_to_enum[results[i]] == 2:
            running_times[i] = running_times[i] * par

    if filter_misclassified:
        verified_indices = np.where(np.array(results) != "Status.Skipped")
        if frequency:
            for checkpoint, checkpoint_features in features.items():
                features[checkpoint] = checkpoint_features[verified_indices]
        else:
            features = features[verified_indices]
        running_times = np.array(running_times)[verified_indices]
        results = np.array(results)[verified_indices]

    enum_results = [result_to_enum[result] for result in results]

    # log10 running times for easier approximation
    running_times = np.log10(running_times)

    # Replace -np.inf or np.inf values (non-existent features) with 0
    if frequency:
        for checkpoint, checkpoint_features in features.items():
            checkpoint_features[checkpoint_features == -np.inf] = 0
            checkpoint_features[checkpoint_features == np.inf] = 0
    else:
        features[features == -np.inf] = 0
        features[features == np.inf] = 0

    return features, running_times, results, np.array(enum_results)


def load_oval_bab_data(log_path, artificial_cutoff=None, fixed_timeout=None, par=1, features_from_log=True,
                       feature_path=None, no_classes=10,
                       neuron_count=0, feature_collection_cutoff=10, filter_misclassified=False, frequency=None):
    """
    Function to return features, running times and verification results based on a log file of Oval.
    :param log_path: path to log file
    :param artificial_cutoff: cutoff time for verification, may be lower than actual cutoff time
    :param par: PAR score, i.e. factor by which the running times of unsolved instances are penalised
    :param features_from_log: if features should be parsed from the log file or if they should be loaded from a numpy array
    :param feature_path: must be provided if features_from_log is False. Path to numpy array of features.
    :param neuron_count: neuron count of verified network
    :param no_classes: number of output classes of verified network
    :param feature_collection_cutoff: seconds for which features should be collected
    :param filter_misclassified: if misclassified instances should be included or not
    :param frequency: frequencies for which features should be provided
    :return: features, running times, verification results in clear text and verification results as enum.
        If frequency is not None, features are a dict where features are provided for each checkpoint, else a numpy array.
    """
    running_time_dict = parse_oval_log(load_log_file(log_path))
    if features_from_log:
        features = get_features_from_verification_log_oval(load_log_file(log_path),
                                                           bab_feature_cutoff=feature_collection_cutoff,
                                                           total_neuron_count=neuron_count,
                                                           frequency=frequency)
    else:
        features = np.load(feature_path, allow_pickle=True)
    running_times = [value["time"] for key, value in running_time_dict.items()]
    results = [value["result"] for key, value in running_time_dict.items()]

    for i in range(len(running_times)):
        if artificial_cutoff and running_times[i] > artificial_cutoff:
            running_times[i] = artificial_cutoff
            results[i] = "Timeout"
        if fixed_timeout and results[i] == "Timeout":
            running_times[i] = fixed_timeout
        if results[i] == "Timeout":
            running_times[i] = running_times[i] * par
    if filter_misclassified:
        verified_indices = np.where(np.array(results) != "SKIPPED")

        if frequency:
            for checkpoint, checkpoint_features in features.items():
                features[checkpoint] = checkpoint_features[verified_indices]
        else:
            features = features[verified_indices]

        running_times = np.array(running_times)[verified_indices]
        results = np.array(results)[verified_indices]

    enum_results = [result_to_enum[result] for result in results]

    # log10 running times for easier approximation
    running_times = np.log10(running_times)

    # Replace -np.inf or np.inf values (non-existent features) with 0
    if frequency:
        for checkpoint, checkpoint_features in features.items():
            checkpoint_features[checkpoint_features == -np.inf] = 0
            checkpoint_features[checkpoint_features == np.inf] = 0
    else:
        features[features == -np.inf] = 0
        features[features == np.inf] = 0

    return features, running_times, results, np.array(enum_results)
