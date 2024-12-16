from experiments.algorithm_selection.config import CONFIG_ADAPTIVE_ALGORITHM_SELECTION, \
    CONFIG_ADAPTIVE_ALGORITHM_SELECTION_AND_TERMINATION
from experiments.running_time_prediction.config import CONFIG_RUNNING_TIME_REGRESSION, CONFIG_TIMEOUT_CLASSIFICATION, \
    CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION
from src.util.tables import create_running_time_regression_table, create_timeouts_table, \
    create_timeout_termination_table, create_algorithm_selection_table, create_benchmark_overview_table

table_running_time_regression = create_running_time_regression_table(
    results_path=CONFIG_RUNNING_TIME_REGRESSION["RESULTS_PATH"])
with open("./tables/table_running_time_regression.csv", 'w', encoding='u8') as f:
    f.write(table_running_time_regression)

table_timeouts_fixed_feature_collection = create_timeouts_table(
    results_path=CONFIG_TIMEOUT_CLASSIFICATION["RESULTS_PATH"],
    thresholds=[.5, .99]
)

with open("./tables/table_timeouts_fixed_feature_collection.csv", 'w', encoding='u8') as f:
    f.write(table_timeouts_fixed_feature_collection)


for thresh in [.5, .8, .9, .99]:
    table_timeouts_continuous_feature_collection = create_timeouts_table(
        results_path=CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION["RESULTS_PATH"],
        thresholds=[thresh]
    )

    with open(f"./tables/table_timeouts_continuous_feature_collection_{thresh}.csv", 'w', encoding='u8') as f:
        f.write(table_timeouts_continuous_feature_collection)

    table_timeout_termination = create_timeout_termination_table(
        results_path=CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION["RESULTS_PATH"],
        thresholds=[thresh]
    )

    with open(f"./tables/table_timeout_termination_{thresh}.csv", 'w', encoding='u8') as f:
        f.write(table_timeout_termination)


table_algorithm_selection = create_algorithm_selection_table(
    results_path=CONFIG_ADAPTIVE_ALGORITHM_SELECTION["RESULTS_PATH"],
    thresholds=[.5, .99]
)

with open(f"./tables/table_algorithm_selection.csv", 'w', encoding='u8') as f:
    f.write(table_algorithm_selection)

table_algorithm_selection_with_termination = create_algorithm_selection_table(
    results_path=CONFIG_ADAPTIVE_ALGORITHM_SELECTION_AND_TERMINATION["RESULTS_PATH"],
    thresholds=[.5, .99]
)

with open(f"./tables/table_algorithm_selection_and_termination.csv", 'w', encoding='u8') as f:
    f.write(table_algorithm_selection_with_termination)

table_benchmark_overview = create_benchmark_overview_table(
    results_path=CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION["RESULTS_PATH"],
)

with open(f"./tables/benchmark_overview.csv", 'w', encoding='u8') as f:
    f.write(table_benchmark_overview)

