import shap
from matplotlib import pyplot as plt
from shap.plots import beeswarm
import numpy as np

from src.util.constants import ABCROWN_FEATURE_NAMES, OVAL_FEATURE_NAMES, VERINET_FEATURE_NAMES


def get_shapley_explanation(rf_model, X_test, X_train, test_running_times, feature_collection_cutoff, results_path, fold, checkpoint=None):
    unsolved_instances = np.where(test_running_times > feature_collection_cutoff)[0]
    if len(unsolved_instances) == 0:
        return [], []
    X_test = X_test[unsolved_instances]
    # TODO: THIS IS A HACK!!!
    if "ABCROWN" in results_path:
        feature_names = ABCROWN_FEATURE_NAMES
    elif "VERINET" in results_path:
        feature_names = VERINET_FEATURE_NAMES
    elif "OVAL-BAB" in results_path:
        feature_names = OVAL_FEATURE_NAMES
    explainer = shap.TreeExplainer(rf_model, model_output="raw", feature_names=feature_names, feature_perturbation="tree_path_dependent")
    explanation = explainer(X_test, check_additivity=True)
    shapley_values = explanation.abs.mean(0).values
    if len(shapley_values.shape) > 1:
        shapley_values = shapley_values[:, 1]
        shapley_values_per_instance = explanation[:, :, 1]
    else:
        shapley_values_per_instance = explanation[:, :]
    shapley_dict = {
        feature: shap_value
        for feature, shap_value in zip(feature_names, shapley_values)
    }
    if checkpoint is None or checkpoint % 60 == 0 or checkpoint == 10:
        beeswarm(
            # only choose explanation for TIMEOUT class (which is symmetrical with no timeout class)
            shap_values=shapley_values_per_instance,
            show=False,
            max_display=100,
            plot_size=(25, 25)
        )
        plt.savefig(results_path + f'/shapley_explanations_{fold}.png')
        plt.close()
    return shapley_dict, shapley_values_per_instance
