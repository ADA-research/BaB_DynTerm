import json
import os

import sklearn.preprocessing
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import seaborn as sns

from src.parsers.parse_verinet_log import parse_verinet_log


def create_scatter_plot(predicted_runtimes, real_runtimes, satisfiability_labels=None,
                        x_label="True running time [GPU s]",
                        y_label="Predicted running time [GPU s]",
                        filename="scatter_plot_test_predictions.png", feature_collection_cutoff=None, legend="full",
                        min_value=10**-3, max_value=10**3):
    """
    Creates scatter plot of predicted running times against true running times.
    :param predicted_runtimes: array of predictions
    :param real_runtimes: array of true values
    :param satisfiability_labels: array of verification results
    :param x_label: label of x axis
    :param y_label: label of y axis
    :param filename: filename to store scatter plot to
    :param feature_collection_cutoff: seconds after which running time prediction was made
    :param legend: legend option for seaborn scatter plot
    :param min_value: min x/y value
    :param max_value:  max x/y value
    """
    if satisfiability_labels is None:
        satisfiability_labels = []

    predicted_runtimes = predicted_runtimes.flatten()
    print(len(predicted_runtimes))

    predicted_runtimes = np.power(10, predicted_runtimes)
    real_runtimes = np.power(10, real_runtimes)

    sns.set(font_scale=3)
    sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})
    plt.figure(figsize=(16, 16))

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


def create_confusion_matrix(predictions, labels, filename="./confusion_matrix.png"):
    """
    Creates Confusion matrix given class labels and predictions.
    :param predictions: class predictions
    :param labels: class labels
    :param filename: filename to store confusion matrix to
    """
    disp = ConfusionMatrixDisplay.from_predictions(labels, predictions)
    title = "Timeout Prediction"
    acc = accuracy_score(labels, predictions)
    title += f'\n Acc: {acc * 100:.2f} %'
    disp.ax_.set_title(title)
    plt.savefig(filename)
    plt.close()


def create_ecdf_plot(running_times_all_verifiers, results_all_verifiers, filename):
    """
    Creates an ECDF plot of running times of different verifiers
    :param running_times_all_verifiers: dict containing running times of all verifiers
    :param results_all_verifiers: dict containing verification results of all verifiers
    :param filename: filename to save plot to
    """
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


