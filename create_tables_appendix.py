from experiments.running_time_prediction.config import CONFIG_RUNNING_TIME_REGRESSION, CONFIG_TIMEOUT_CLASSIFICATION, \
    CONFIG_DYNAMIC_ALGORITHM_TERMINATION
from src.util.tables import create_running_time_regression_table, create_timeouts_table, \
    create_timeout_termination_table, create_benchmark_overview_table

table_running_time_regression = create_running_time_regression_table(
    results_path=CONFIG_RUNNING_TIME_REGRESSION["RESULTS_PATH"])
with open("./tables/table_running_time_regression.csv", 'w', encoding='u8') as f:
    f.write(table_running_time_regression)

for thresh in [.5, .9,]:
    table_timeouts_continuous_feature_collection = create_timeouts_table(
        results_path=CONFIG_DYNAMIC_ALGORITHM_TERMINATION["RESULTS_PATH"],
        thresholds=[thresh]
    )

    with open(f"./tables/table_timeouts_continuous_feature_collection_{thresh}.csv", 'w', encoding='u8') as f:
        f.write(table_timeouts_continuous_feature_collection)

    table_timeout_termination = create_timeout_termination_table(
        results_path=CONFIG_DYNAMIC_ALGORITHM_TERMINATION["RESULTS_PATH"],
        thresholds=[thresh]
    )

    with open(f"./tables/table_timeout_termination_{thresh}.csv", 'w', encoding='u8') as f:
        f.write(table_timeout_termination)

