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


def load_algorithm_selection_data(abcrown_log_file, verinet_log_file, oval_log_file, feature_collection_cutoff=None,
                                  frequency=None,
                                  neuron_count=None, cutoff=None, par=None, filter_misclassified=True,
                                  no_classes=10):
    """
    Function to gather features, running_times, verification results and labels that assign each instance the fastest verifier
    for a benchmark.
    :param abcrown_log_file: location of ab-CROWN log file.
    :param verinet_log_file: location of VeriNet log file.
    :param oval_log_file: location of Oval log file.
    :param feature_collection_cutoff: seconds until features should be collected
    :param frequency: frequency for which features should be provided
    :param neuron_count: neuron count of verified network
    :param cutoff: maximum running time of verification procedure
    :param par: par score for running times, i.e. factor by which running times of unsolved instances are penalised.
    :param filter_misclassified: if misclassified instances should be included or not.
    :param no_classes: number of output classes of the verified neural network.
    :return: concatenated features of all verifiers for each instance and verifier, running times for each instance and verifier,
        verification results of each instance and verifier (as enum and as literals),
        features for best_verifier prediction case, i.e. where each instance gets only one feature item that is the concatenation of
        all verifier features, labels that assign each instance the fastest verification tool and verifier_data, i.e.
        the bare running times and verification results of each verifier.s
    """
    verifier_data = defaultdict(dict)
    if abcrown_log_file:
        abcrown_features, abcrown_running_times, abcrown_results, abcrown_enum_results = load_abcrown_data(
            log_path=abcrown_log_file,
            artificial_cutoff=cutoff,
            par=par,
            neuron_count=neuron_count,
            feature_collection_cutoff=feature_collection_cutoff,
            frequency=frequency,
            filter_misclassified=False
        )
        verifier_data[ABCROWN]["no_features"] = abcrown_features[frequency].shape[1]
        verifier_data[ABCROWN]["features"] = abcrown_features
        verifier_data[ABCROWN]["running_times"] = abcrown_running_times
        verifier_data[ABCROWN]["results"] = abcrown_results
        verifier_data[ABCROWN]["enum_results"] = abcrown_enum_results

    if verinet_log_file:
        verinet_features, verinet_running_times, verinet_results, verinet_enum_results = load_verinet_data(
            log_path=verinet_log_file,
            artificial_cutoff=cutoff,
            par=par,
            neuron_count=neuron_count,
            feature_collection_cutoff=feature_collection_cutoff,
            frequency=frequency,
            filter_misclassified=False,
            no_classes=no_classes
        )
        verifier_data[VERINET]["no_features"] = verinet_features[frequency].shape[1]
        verifier_data[VERINET]["features"] = verinet_features
        verifier_data[VERINET]["running_times"] = verinet_running_times
        verifier_data[VERINET]["results"] = verinet_results
        verifier_data[VERINET]["enum_results"] = verinet_enum_results

    if oval_log_file:
        oval_features, oval_running_times, oval_results, oval_enum_results = load_oval_bab_data(
            log_path=oval_log_file,
            artificial_cutoff=cutoff,
            par=par,
            neuron_count=neuron_count,
            feature_collection_cutoff=feature_collection_cutoff,
            frequency=frequency,
            filter_misclassified=False
        )
        verifier_data[OVAL]["no_features"] = oval_features[frequency].shape[1]
        verifier_data[OVAL]["features"] = oval_features
        verifier_data[OVAL]["running_times"] = oval_running_times
        verifier_data[OVAL]["results"] = oval_results
        verifier_data[OVAL]["enum_results"] = oval_enum_results

    # check that each verifier has the same number of instances
    if not frequency:
        no_instances = list(set([len(verifier_data[verifier]["features"]) for verifier in verifier_data]))
    else:
        no_instances = list(
            set([len(verifier_data[verifier]["features"][checkpoint]) for verifier in verifier_data for checkpoint in
                 verifier_data[verifier]["features"]]))
    assert len(no_instances) == 1

    no_instances = no_instances[0]
    no_verifiers = len(verifier_data.keys())

    # merge running_times, results, and results_to_enum
    running_times = []
    results = []
    enum_results = []

    # filter misclassified instances
    # we have to do it in this way, as abCROWN has no way to detect misclassifications during runtime
    misclassified_indices = []
    if filter_misclassified:
        for index in range(no_instances):
            index_results = set([verifier_data[verifier]["results"][index] for verifier in verifier_data])
            if "Status.Skipped" in index_results or "SKIPPED" in index_results:
                misclassified_indices.append(index)

    for index in range(no_instances):
        if index in misclassified_indices:
            continue
        for verifier in verifier_data:
            running_times.append(verifier_data[verifier]["running_times"][index])
            results.append(verifier_data[verifier]["results"][index])
            enum_results.append(verifier_data[verifier]["enum_results"][index])

    # merge features of all verifiers together
    if frequency:
        if cutoff:
            max_time = cutoff
        else:
            max_time = max([max(verifier_data[verifier]["features"].keys()) for verifier in verifier_data])

        features = {
            checkpoint: [np.array([]) for _ in range(no_instances * no_verifiers)]
            for checkpoint in range(frequency, max_time, frequency)
        }
    else:
        features = [np.array([]) for _ in range(no_instances * no_verifiers)]

    for feature_index in range(no_instances):
        if feature_index in misclassified_indices:
            continue
        if frequency:
            for checkpoint in range(frequency, max_time, frequency):
                merged_feature = np.array([])
                for verifier in verifier_data:
                    merged_feature = np.append(merged_feature,
                                               verifier_data[verifier]["features"][checkpoint][feature_index])
                for i in range(no_verifiers):
                    index = feature_index * no_verifiers
                    features[checkpoint][index + i] = np.append(merged_feature, i)

        else:
            merged_feature = np.array([])
            for verifier in verifier_data:
                merged_feature = np.append(merged_feature, verifier_data[verifier]["features"][feature_index])
            for i in range(no_verifiers):
                index = feature_index * no_verifiers
                features[index + i] = np.append(merged_feature, i)

    # remove empty arrays (misclassified indices)
    if frequency:
        for checkpoint in range(frequency, max_time, frequency):
            features[checkpoint] = [feature for feature in features[checkpoint] if feature.size > 0]
    else:
        features = [feature for feature in features if feature.size > 0]

    best_verifiers = []
    if frequency:
        features_best_verifiers = {
            checkpoint: []
            for checkpoint in range(frequency, max_time, frequency)
        }
    else:
        features_best_verifiers = []
    for index in range(int(len(running_times) / no_verifiers)):
        instance_index = index * no_verifiers
        if set(enum_results[instance_index:instance_index + no_verifiers]) == {2}:
            best_verifier = -1
        else:
            best_verifier = np.argmin(running_times[instance_index:instance_index + no_verifiers])
        best_verifiers.append(best_verifier)
        if frequency:
            for checkpoint in range(frequency, max_time, frequency):
                features_best_verifiers[checkpoint].append(features[checkpoint][instance_index][:-1])
        else:
            features_best_verifiers.append(
                features[instance_index][:-1])  # we don't need the verifier index in this case

    if frequency:
        for checkpoint in features:
            features[checkpoint] = np.array(features[checkpoint])
            features_best_verifiers[checkpoint] = np.array(features_best_verifiers[checkpoint])
    else:
        features = np.array(features)
        features_best_verifiers = np.array(features_best_verifiers)

    return features, np.array(running_times), np.array(results), np.array(
        enum_results), features_best_verifiers, np.array(best_verifiers), verifier_data
