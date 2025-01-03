from experiments.running_time_prediction.config import CONFIG_RUNNING_TIME_REGRESSION, CONFIG_TIMEOUT_CLASSIFICATION, \
    CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION
from src.util.tables import create_timeouts_table, \
    create_timeout_termination_table, create_benchmark_overview_table

if __name__ == "__main__":

    thresh = .99

    table_timeouts_continuous_feature_collection = create_timeouts_table(
        results_path=CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION["RESULTS_PATH"],
        thresholds=[thresh]
    )

    with open(f"./tables/table_metrics_adaptive_timeout_classification_{thresh}.csv", 'w', encoding='u8') as f:
        f.write(table_timeouts_continuous_feature_collection)

    table_timeout_termination = create_timeout_termination_table(
        results_path=CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION["RESULTS_PATH"],
        thresholds=[thresh]
    )

    with open(f"./tables/table_dynamic_timeout_termination_{thresh}.csv", 'w', encoding='u8') as f:
        f.write(table_timeout_termination)

    table_benchmark_overview = create_benchmark_overview_table(
        results_path=CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION["RESULTS_PATH"],
    )

    with open(f"./tables/benchmark_overview.csv", 'w', encoding='u8') as f:
        f.write(table_benchmark_overview)

