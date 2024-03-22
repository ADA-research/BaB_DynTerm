from experiments.running_time_prediction.config import CONFIG_RUNNING_TIME_REGRESSION, CONFIG_TIMEOUT_CLASSIFICATION, \
    CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION
from src.util.tables import create_running_time_table, create_timeouts_table, create_timeout_termination_table

table_running_time_regression = create_running_time_table(
    results_path=CONFIG_RUNNING_TIME_REGRESSION["RESULTS_PATH"]
)
with open("./tables/table_running_time_regression.csv", 'w', encoding='u8') as f:
    f.write(table_running_time_regression)

table_timeouts_fixed_feature_collection = create_timeouts_table(
    results_path=CONFIG_TIMEOUT_CLASSIFICATION["RESULTS_PATH"],
    thresholds=[.5, .99]
)

with open("./tables/table_timeouts_fixed_feature_collection.csv", 'w', encoding='u8') as f:
    f.write(table_timeouts_fixed_feature_collection)

table_timeouts_continuous_feature_collection = create_timeouts_table(
    results_path=CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION["RESULTS_PATH"],
    thresholds=[.5, .99]
)

with open("./tables/table_timeouts_continuous_feature_collection.csv", 'w', encoding='u8') as f:
    f.write(table_timeouts_continuous_feature_collection)

table_timeout_termination = create_timeout_termination_table(
    results_path=CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION["RESULTS_PATH"],
    thresholds=[.99]
)

with open("./tables/table_timeout_termination.csv", 'w', encoding='u8') as f:
    f.write(table_timeout_termination)




