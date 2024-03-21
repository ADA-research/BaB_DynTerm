import json
import os

import sklearn.preprocessing
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import seaborn as sns

from src.parsers.parse_ab_crown_log import parse_abcrown_log
from src.parsers.parse_verinet_log import parse_verinet_log
from src.util.io import load_log_file
from src.util.constants import result_to_enum


def draw_training_data_histograms(log_path, plot_title, artificial_cutoff=None, load_data_func=parse_abcrown_log):
    running_time_dict = load_data_func(load_log_file(log_path))
    running_time_dict = {key: item for key, item in running_time_dict.items() if item["result"] != "Status.Skipped"}
    benchmark_data_sat = []
    benchmark_data_unsat = []
    benchmark_data_timeout = []
    for problem_id in running_time_dict:
        data = running_time_dict[problem_id]
        result = result_to_enum[data["result"]]
        time_needed = data["time"]
        if artificial_cutoff and artificial_cutoff < time_needed:
            time_needed = artificial_cutoff
            result = result_to_enum["unknown"]
        if result == 0:
            benchmark_data_sat.append(time_needed)
        elif result == 1:
            benchmark_data_unsat.append(time_needed)
        else:
            benchmark_data_timeout.append(time_needed)
    plt.hist((benchmark_data_unsat, benchmark_data_sat, benchmark_data_timeout), bins=20,
             label=["unsat", "sat", "timeout"])
    plt.legend()
    plt.title(plot_title)
    plt.savefig(f"./{plot_title}_runtimes_hist.png")
    plt.close()


def draw_all_running_time_histograms(results_path):
    for benchmark in os.listdir(results_path):
        benchmark_name = benchmark.replace("_", " ")
        results_path_benchmark = f"{results_path}/{benchmark}"
        for filename in os.listdir(results_path_benchmark):
            if "metrics" in filename or os.path.isdir(f"{results_path_benchmark}/{filename}") \
                    or filename.split(".")[-1] != "json":
                continue

            verifier = filename.split(".")[0]

            with open(f"{results_path_benchmark}/{filename}", 'r') as f:
                verifier_data = json.load(f)

            draw_training_data_histograms(verifier_data, f"{benchmark_name} {verifier}",
                                          f"{results_path_benchmark}/{benchmark_name}_{verifier}")


def draw_normalized_feature_boxplots(feature_path, plot_title, feature_names=None):
    features = np.load(feature_path)
    features[features == -np.inf] = 0
    normalized_features = sklearn.preprocessing.StandardScaler().fit_transform(features)

    plt.figure(figsize=(15, 6))
    plt.boxplot(normalized_features, vert=True, patch_artist=True)

    # Customize the plot
    plt.title(f'Boxplots for {plot_title}')
    plt.xlabel('Feature')
    plt.ylabel('Values')
    plt.xticks(np.arange(1, 10),
               [f'{feature_names[i] if feature_names else i}' for i in range(normalized_features.shape[1])])

    # Show the plot
    plt.savefig(f"./{plot_title}_feature_boxplots.png")
    plt.close()


def draw_impurity_based_feature_importance(forest_classifier, fold):
    importances = forest_classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest_classifier.estimators_], axis=0)
    forest_importances = pd.Series(importances, index=[str(i) for i in range(importances.shape[0])])

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig(f'feature-importance-fold-{fold}.png')


def create_scatter_plot(predicted_runtimes, real_runtimes, satisfiability_labels=None,
                        x_label="True running time [GPU s]",
                        y_label="Predicted running time [GPU s]",
                        filename="scatter_plot_test_predictions.png", feature_collection_cutoff=None, legend="full"):

    if satisfiability_labels is None:
        satisfiability_labels = []

    predicted_runtimes = predicted_runtimes.flatten()
    print(len(predicted_runtimes))

    predicted_runtimes = np.power(10, predicted_runtimes)
    real_runtimes = np.power(10, real_runtimes)

    sns.set(font_scale=3)
    sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})
    plt.figure(figsize=(16, 16))

    # min_value = min(min(real_runtimes), min(predicted_runtimes))
    # max_value = max(max(real_runtimes), max(predicted_runtimes))
    min_value = 10 ** -3
    max_value = 10 ** 3

    # Plot the ideal line (where predicted = real)
    plt.plot([min_value, max_value], [min_value, max_value], color='green', linestyle='--', label='Ideal')
    if feature_collection_cutoff:
        plt.axvline(x=pow(10, feature_collection_cutoff), color="red", linestyle="--", linewidth=2,
                    label="Feature Cutoff")

    if len(satisfiability_labels) == 0:
        ax = sns.scatterplot(x=real_runtimes, y=predicted_runtimes)
    else:
        palette = {"UNSAT": "blue", "SAT": "orange", "Timeout": "green"}
        sat_indices, = np.where(satisfiability_labels == 0)
        unsat_indices, = np.where(satisfiability_labels == 1)
        timeout_indices, = np.where(satisfiability_labels == 2)
        data = {
            "real_runtimes": real_runtimes,
            "predicted_runtimes": predicted_runtimes,
            "Legend": [
                "UNSAT" if idx in unsat_indices else
                ("SAT" if idx in sat_indices else "Timeout")
                for idx in range(len(real_runtimes))
            ]
        }
        df = pd.DataFrame(data)
        ax = sns.scatterplot(
            x="real_runtimes",
            y="predicted_runtimes",
            hue="Legend",
            style="Legend",
            hue_order=["UNSAT", "SAT", "Timeout"],
            palette=palette,
            data=df,
            legend=legend
        )
        plt.plot(linewidth=1)
        ax.set(xscale="log", yscale="log")
        ax.set(xlabel=x_label, ylabel=y_label)
        ax.set_xlim(10 ** (-3), 10 ** 3)  # Set x-axis limits
        ax.set_ylim(10 ** (-3), 10 ** 3)  # Set y-axis limits
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # # Add labels and title
    # ax.set_xlabel(x_label)
    # ax.set_ylabel(y_label)
    # ax.set_title(title)
    # ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    # plt.tight_layout()

    # Display the plot
    plt.savefig(filename)


def create_confusion_matrix(predictions, labels, incompletes_included, filename="./confusion_matrix.png"):
    disp = ConfusionMatrixDisplay.from_predictions(labels, predictions)
    title = "Timeout Prediction"
    if incompletes_included:
        title += " - Incompletes Included"
    else:
        title += ' - Incompletes Excluded'
    acc = accuracy_score(labels, predictions)
    title += f'\n Acc: {acc * 100:.2f} %'
    disp.ax_.set_title(title)
    plt.savefig(filename)
    plt.close()


def create_ecdf_plot(running_times_all_verifiers, results_all_verifiers, filename):
    sns.set(style="whitegrid")
    sns.set_palette("colorblind")
    sns.set(font_scale=3)
    sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})
    plt.figure(figsize=(20, 20))

    max_time_taken = 0

    for verifier, running_times in running_times_all_verifiers.items():
        results = results_all_verifiers[verifier]

        # HACK: we set timeouts to very high time and later adjust the range of the x axis s.t. plot is correct!
        running_times_plot = []
        time_taken = 0
        unsolved = 0
        for running_time, result in zip(running_times, results):
            time_taken += pow(10, running_time)
            if result not in [2, None]:
                running_times_plot.append(time_taken)
            else:
                unsolved += 1

        max_time_taken = max([max_time_taken, np.log10(time_taken)])
        for i in range(unsolved):
            running_times_plot = np.append(running_times_plot, [10000000000])
        ax = sns.ecdfplot(data=running_times_plot, label=verifier, linewidth=2, stat="proportion",
                          log_scale=[True, False])

    plt.xlim(10 ** -0.0000001, pow(10, max_time_taken))
    plt.xlabel('Running Time [GPU s]')
    plt.ylabel('Proportion of Solved Instances')
    plt.legend()

    plt.savefig(filename)
    plt.close()
    sns.reset_defaults()

    with open(f"{filename}.json", 'w') as f:
        ecdf_data = {
            "results": {verifier: result.tolist() if isinstance(result, ndarray) else result for verifier, result in
                        results_all_verifiers.items()},
            "running_times": {verifier: running_time.tolist() if isinstance(running_time, ndarray) else running_time for
                              verifier, running_time in running_times_all_verifiers.items()}
        }
        json.dump(ecdf_data, f, indent=2)


if __name__ == "__main__":
    draw_training_data_histograms("./verification_logs/VeriNet/CIFAR_RESNET_2B/VERINET-CIFAR-RESNET2B-40843902.log",
                                  "CIFAR-10 Resnet 2B Verinet", load_data_func=parse_verinet_log)
