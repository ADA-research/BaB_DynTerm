import json
import os

import sklearn.preprocessing
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker
from numpy import ndarray
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import seaborn as sns

from experiments.running_time_prediction.config import CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION
from src.parsers.parse_verinet_log import parse_verinet_log
from src.util.constants import experiment_groups, experiment_samples, SUPPORTED_VERIFIERS, VERIFIER_TO_TEX, \
    ALL_EXPERIMENTS


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
    plt.close()


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


def plot_performance_against_theta(config):
    results_path = config["RESULTS_PATH"]
    for verifier in SUPPORTED_VERIFIERS:
        plt_data_wct = []
        plt_data_solved = []
        print(f"------------------- {verifier} ---------------")
        for thresh in config["TIMEOUT_CLASSIFICATION_THRESHOLDS"]:
            avg_solved = 0
            avg_wct = 0
            no_observations = 0
            for experiment_group, experiments in experiment_groups.items():
                for experiment in experiments:
                    experiment_path = f"{results_path}/{experiment}"
                    no_instances = experiment_samples[experiment]
                    verifier_results_path = f"./{experiment_path}/{verifier}"
                    if not os.path.exists(f"{verifier_results_path}/ecdf_threshold_{thresh}.png.json"):
                        continue
                    with open(f"{verifier_results_path}/ecdf_threshold_{thresh}.png.json", 'r') as f:
                        verifier_data = json.load(f)
                    vanilla_results = verifier_data["results"]["Vanilla Verifier"]
                    # one of the hustles one has to do because abCROWN did not filter misclassified instances
                    if no_instances != len(vanilla_results):
                        misclassified = len(vanilla_results) - no_instances
                        timeout_termination_running_times = verifier_data["running_times"]["Timeout Prediction"]
                        timeout_termination_results = verifier_data["results"]["Timeout Prediction"]
                        no_timeouts_timeout_termination = sum(
                            [1 for result in timeout_termination_results if result == 2.])
                        no_solved_timeout_termination = sum(
                            [1 for result in timeout_termination_results if result != 2.]) - misclassified
                    else:
                        timeout_termination_running_times = verifier_data["running_times"]["Timeout Prediction"]
                        timeout_termination_results = verifier_data["results"]["Timeout Prediction"]
                        no_timeouts_timeout_termination = sum(
                            [1 for result in timeout_termination_results if result == 2.])
                        no_solved_timeout_termination = sum(
                            [1 for result in timeout_termination_results if result != 2.])

                    wct_timeout_termination = sum(
                        [pow(10, log_running_time) for log_running_time in timeout_termination_running_times])

                    avg_wct += wct_timeout_termination
                    avg_solved += no_solved_timeout_termination
                    no_observations += 1

            avg_solved /= no_observations
            avg_wct /= no_observations
            plt_data_wct.append(avg_wct / 60 / 60)
            plt_data_solved.append(avg_solved)
            print(f"THETA {thresh}: AVG SOLVED {avg_solved} / AVG WCT {avg_wct}")

        # sns.set(style="white")
        # sns.set_palette("colorblind")
        sns.set(font_scale=1, style="white")
        sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})
        plt.figure(figsize=(30, 30))
        fig, ax = plt.subplots()
        ax2 = ax.twinx()  # Create a second axes sharing the x-axis
        no_solved_plot = ax.plot(list(config["TIMEOUT_CLASSIFICATION_THRESHOLDS"]), plt_data_solved, 'o-', linewidth=2, color="red", label="Avg. # Solved Instances")
        ax.set_ylabel("# Solved Instances")
        wct_plot = ax2.plot(list(config["TIMEOUT_CLASSIFICATION_THRESHOLDS"]), plt_data_wct, 'o-', linewidth=2, color='blue', label="Avg. Running Time")
        ax2.set_ylabel("Running Time [GPU h]")
        ax.set_xlabel(r"$\theta$")
        # ax2.set_ylim([min(plt_data_solved) // 10 * 10, max(plt_data_solved) // 10 * 10 + 10])
        # ax.set_ylim([min(plt_data_wct) // 1, max(plt_data_wct) // 1 + 1])
        lns = wct_plot + no_solved_plot
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc="upper left")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{config['RESULTS_PATH']}/theta_distribution_{verifier}.pdf")


def create_ecdf_for_presentation():
    """
    Creates an ECDF plot of running times of different verifiers
    :param running_times_all_verifiers: dict containing running times of all verifiers
    :param results_all_verifiers: dict containing verification results of all verifiers
    :param filename: filename to save plot to
    """

    benchmark_to_tex = {
        "MNIST_9_100": "MNIST 8 100",
        "CIFAR_RESNET_2B": "CIFAR-10 ResNet 2B",
        "VIT": "CIFAR-10 Vision Transformer",
        "TINY_IMAGENET": "Tiny ImageNet ResNet Med"
    }

    for benchmark in ['TINY_IMAGENET', 'MNIST_9_100', "CIFAR_RESNET_2B", "VIT"]:

        for verifier in ['ABCROWN']:
            if not os.path.exists(f"./results/results_continuous_timeout_classification/{benchmark}/{verifier}/ecdf_threshold_0.99.png.json"):
                continue
            with open(f'./results/results_continuous_timeout_classification/{benchmark}/{verifier}/ecdf_threshold_0.99.png.json', "r") as f:
                data = json.load(f)

            our_results = data['results']['Timeout Prediction']
            vanilla_results = data['results']['Vanilla Verifier']
            our_running_times = data['running_times']['Timeout Prediction']
            vanilla_running_times = data['running_times']['Vanilla Verifier']

            sns.set(style="whitegrid")
            sns.set_palette("dark")
            sns.set(font_scale=3)
            sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})
            plt.figure(figsize=(20, 20))
            max_time_taken = 0

            running_times_plot = []
            time_taken = 0
            unsolved = 0

            # Compute the cumulative times
            for running_time, result in zip(vanilla_running_times, vanilla_results):
                time_taken += pow(10, running_time)
                if result not in [2, None]:
                    running_times_plot.append(time_taken)
                else:
                    unsolved += 1

            max_time_taken = max([max_time_taken, np.log10(time_taken)])

            # Sort running times and compute cumulative proportion
            running_times_plot = np.sort(running_times_plot)
            cumulative_proportion = np.arange(1, len(running_times_plot) + 1) / (len(running_times_plot) + unsolved)

            # Plot using sns.lineplot
            ax = sns.lineplot(x=running_times_plot, y=cumulative_proportion,
                              label=f"{VERIFIER_TO_TEX[verifier]}", linewidth=5)

            line_color = ax.lines[-1].get_color()  # Get the last line's color

            plt.scatter(running_times_plot[-1], cumulative_proportion[-1], s=200, marker='s', color=line_color)

            running_times_plot = []
            time_taken = 0
            unsolved = 0

            # Compute the cumulative times
            for running_time, result in zip(our_running_times, our_results):
                time_taken += pow(10, running_time)
                if result not in [2, None]:
                    running_times_plot.append(time_taken)
                else:
                    unsolved += 1

            max_time_taken = max([max_time_taken, np.log10(time_taken)])

            # Sort running times and compute cumulative proportion
            running_times_plot = np.sort(running_times_plot)
            cumulative_proportion = np.arange(1, len(running_times_plot) + 1) / (len(running_times_plot) + unsolved)

            # Plot using sns.lineplot
            ax = sns.lineplot(x=running_times_plot, y=cumulative_proportion,
                              label=f"{VERIFIER_TO_TEX[verifier]} w/ Dynamic Termination", linewidth=5)
            line_color = ax.lines[-1].get_color()  # Get the last line's color

            plt.scatter(running_times_plot[-1], cumulative_proportion[-1], s=200, marker='s', color=line_color)

            plt.xlim(10 ** -0.0000001, pow(10, max_time_taken + .1))
            plt.ylim(0, 1)
            plt.xscale("symlog", linthresh=1000)
            plt.xlabel('Total Running Time [GPU s]')
            plt.ylabel('Proportion of Solved Instances')
            plt.legend(loc="upper left")

            ax.set_title(f"Benchmark: {benchmark_to_tex[benchmark]}", pad=20)

            plt.tight_layout()

            plt.savefig(f"./ECDF_{benchmark}_{verifier}.pdf")
            plt.close()
            sns.reset_defaults()

def create_bar_plots_presentation():
    # Sample data
    benchmarks = ALL_EXPERIMENTS
    tools = SUPPORTED_VERIFIERS
    metrics = ['precision', 'recall', 'test_acc',]

    sns.set(style="whitegrid")
    sns.set(font_scale=3)
    sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})
    plt.figure(figsize=(20, 30))

    metric_to_tex = {
        "test_acc": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
    }

    data_list = []
    for benchmark in benchmarks:
        for verifier in tools:
            if not os.path.exists(f"./results/results_continuous_timeout_classification/{benchmark}/{verifier}/metrics_thresh_0.99.json"):
                continue
            with open(f"./results/results_continuous_timeout_classification/{benchmark}/{verifier}/metrics_thresh_0.99.json", "r") as f:
                data = json.load(f)

            data_list = data_list + [{
                'Benchmark': benchmark,
                'Verification Tool': VERIFIER_TO_TEX[verifier],
                "Metric": metric_to_tex[metric],
                'Value': data['avg'][metric]
            } for metric in metrics]
    # data_list = [
    #     {'Benchmark': b, 'Tool': t, 'Metric': m, 'Value': data[i, j, k]}
    #     for i, b in enumerate(benchmarks)
    #     for j, t in enumerate(tools)
    #     for k, m in enumerate(metrics)
    # ]
    df = pd.DataFrame(data_list)
    ax = sns.boxplot(x="Metric", y="Value",
                hue="Verification Tool", palette='deep',
                data=df, linewidth=3, showfliers=True, fliersize=15, flierprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black'},
                fill=True)
    sns.despine(offset=10, trim=True)
    sns.move_legend(ax, loc='lower left')
    # plt.legend(loc="upper right")
    # Adjust layout and display the figure
    plt.tight_layout()
    ax.set(xlabel=None, ylabel=None)
    plt.savefig(f"./boxplot_tab_2.pdf")

if __name__ == "__main__":
    create_ecdf_for_presentation()
    create_bar_plots_presentation()

