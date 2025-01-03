import json
import os
from collections import defaultdict

from src.util.constants import SUPPORTED_VERIFIERS, VERIFIER_FEATURE_MAP, TIMEOUT, ALL_EXPERIMENTS


def eval_feature_ablation_study(feature_ablation_study_folder, threshold=0.5, results_folder=None):
    # todo: change that
    experiments = ALL_EXPERIMENTS
    for verifier in SUPPORTED_VERIFIERS:
        table_csv = "Excluded Feature,"
        table_csv += ",,".join(experiments) + "\n"
        table_csv += "," + "FP,TP," * len(experiments) + "\n"
        verifier_differences = {}

        for verifier_feature in VERIFIER_FEATURE_MAP[verifier]:
            table_csv += f"{verifier_feature},"
            avg_feature_differences = defaultdict(float)
            running_time_differences = 0
            no_solved_differences = 0
            no_experiments = len(experiments)

            for experiment in experiments:
                baseline_folder = f"./{feature_ablation_study_folder}/{experiment}/{verifier}/BASELINE/" if not results_folder else f"{results_folder}/{experiment}/{verifier}"
                feature_differences = defaultdict(float)
                if not os.path.exists(
                        f"./{baseline_folder}/metrics_thresh_{threshold}.json"):
                    table_csv += "-,-,"
                    no_experiments -= 1
                    continue
                with open(f"./{baseline_folder}/metrics_thresh_{threshold}.json", "r") as f:
                    standard_results = json.load(f)
                standard_results = standard_results["sum"]

                with open(f"./{baseline_folder}/ecdf_threshold_{threshold}.png.json",
                          "r") as f:
                    standard_running_times = json.load(f)
                no_solved_standard = len(
                    [result for result in standard_running_times["results"]["Timeout Prediction"] if
                     result != TIMEOUT])
                standard_running_times = standard_running_times["running_times"]["Timeout Prediction"]
                standard_running_time = sum([pow(10, running_time) for running_time in standard_running_times])

                with open(
                        f"./{feature_ablation_study_folder}/{experiment}/{verifier}/{verifier_feature}/metrics_thresh_{threshold}.json",
                        "r") as f:
                    feature_results = json.load(f)

                with open(
                        f"./{feature_ablation_study_folder}/{experiment}/{verifier}/{verifier_feature}/ecdf_threshold_{threshold}.png.json",
                        "r") as f:
                    feature_running_times = json.load(f)
                    no_solved_feature = len(
                        [result for result in feature_running_times["results"]["Timeout Prediction"] if
                         result != TIMEOUT])
                    feature_running_times = feature_running_times["running_times"]["Timeout Prediction"]
                    feature_running_time = sum([pow(10, running_time) for running_time in feature_running_times])

                feature_results = feature_results["sum"]

                for metric in feature_results:
                    avg_feature_differences[metric] += feature_results[metric] - standard_results[metric]
                    feature_differences[metric] = feature_results[metric] - standard_results[metric]

                running_time_differences += feature_running_time - standard_running_time
                no_solved_differences += no_solved_feature - no_solved_standard

                # table_csv += f"{round(no_solved_feature - no_solved_standard, 2)},{round((feature_running_time - standard_running_time) / 60, 2)},"
                table_csv += f"{round(feature_differences['fp'], 2)}, {round(feature_differences['tp'], 2)},"
            for metric in avg_feature_differences:
                avg_feature_differences[metric] /= no_experiments

            verifier_differences[verifier_feature] = avg_feature_differences

            running_time_differences /= no_experiments
            no_solved_differences /= no_experiments
            verifier_differences[verifier_feature]["avg_solved_difference"]  = no_solved_differences
            verifier_differences[verifier_feature]["avg_running_time_difference"] = running_time_differences


            # print(
            #     f"FEATURE DIFFERENCES FOR {verifier_feature} ON {verifier} \n {json.dumps(avg_feature_differences, indent=4)}")
            # print(f"AVG. RUNNING TIME DIFFERENCE (minutes): {running_time_differences / 60}")
            # print(f"AVG DIFF OF SOLVED INSTANCES: {no_solved_differences}")

            table_csv += "\n"

        # print("METRICS PER EXPERIMENT AND FEATURE")
        # print("--------------------------------------------------------------")
        # print(table_csv)
        # print("--------------------------------------------------------------")
        # print("AVERAGE METRICS")
        # print("--------------------------------------------------------------")
        avg_csv = ''
        avg_csv += "Excluded Feature,TP,FP,#Solved,Time(m)\n"
        for feature in verifier_differences:
            avg_csv += f"{feature},  {round(verifier_differences[feature]['tp'], 2)}, {round(verifier_differences[feature]['fp'], 2)}, {round(verifier_differences[feature]['avg_solved_difference'], 2)}, {round(verifier_differences[feature]['avg_running_time_difference'] / 60, 4)}\n"

        with open(f'{feature_ablation_study_folder}/feature_ablation_average_{verifier}.csv', 'w') as f:
            f.write(avg_csv)
