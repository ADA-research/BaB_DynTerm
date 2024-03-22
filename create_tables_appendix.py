from experiments.running_time_prediction.config import CONFIG_RUNNING_TIME_REGRESSION, CONFIG_TIMEOUT_CLASSIFICATION, \
    CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION
from src.util.tables import create_running_time_table, create_timeouts_table, create_timeout_termination_table


table_timeouts_continuous_feature_collection = create_timeouts_table(
    results_path=CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION["RESULTS_PATH"],
    thresholds=[.8, .9]
)

with open("./tables/table_timeouts_continuous_feature_collection_appendix.csv", 'w', encoding='u8') as f:
    f.write(table_timeouts_continuous_feature_collection)

table_timeout_termination = create_timeout_termination_table(
    results_path=CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION["RESULTS_PATH"],
    thresholds=[.8, .9]
)

with open("./tables/table_timeout_termination_appendix.csv", 'w', encoding='u8') as f:
    f.write(table_timeout_termination)




