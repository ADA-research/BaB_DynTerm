import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as pl
from shap import Explanation
from shap.plots import colors, beeswarm
from shap.plots._labels import labels
from shap.plots._utils import convert_ordering, convert_color, sort_inds, get_sort_order, merge_nodes
from shap.utils import safe_isinstance
from shap.utils._exceptions import DimensionError

from src.util.constants import SUPPORTED_VERIFIERS, ABCROWN, ABCROWN_FEATURE_NAMES, VERINET, VERIFIER_TO_TEX, OVAL, \
    FEATURES_TO_TEX


def beeswarm_checkpoint_coloring(shap_values, max_display=10, order=Explanation.abs.mean(0),
             clustering=None, cluster_threshold=0.5, color=None,
             axis_color="#333333", alpha=1, show=True, log_scale=False,
             color_bar=True, s=16, plot_size="auto", color_bar_label=labels["FEATURE_VALUE"]):
    """Create a SHAP beeswarm plot, colored by feature values when they are provided.

    Parameters
    ----------
    shap_values : Explanation
        This is an :class:`.Explanation` object containing a matrix of SHAP values
        (# samples x # features).

    max_display : int
        How many top features to include in the plot (default is 10, or 7 for
        interaction plots).

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot to be customized further
        after it has been created, returning the current axis via plt.gca().

    color_bar : bool
        Whether to draw the color bar (legend).

    s : float
        What size to make the markers. For further information see `s` in ``matplotlib.pyplot.scatter``.

    plot_size : "auto" (default), float, (float, float), or None
        What size to make the plot. By default, the size is auto-scaled based on the
        number of features that are being displayed. Passing a single float will cause
        each row to be that many inches high. Passing a pair of floats will scale the
        plot by that number of inches. If ``None`` is passed, then the size of the
        current figure will be left unchanged.

    Examples
    --------
    See `beeswarm plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html>`_.

    """
    checkpoints = []
    values, features, feature_names = None, None, None
    for fold, shap_values_per_fold in enumerate(shap_values):
        for checkpoint, shap_values_per_checkpoint in enumerate(shap_values_per_fold):
            if len(shap_values_per_checkpoint) == 0:
                continue
            if values is None:
                values = np.empty((0, shap_values[fold][checkpoint].values.shape[1]))
                features = np.empty((0, shap_values[fold][checkpoint].data.shape[1]))
                feature_names = shap_values[fold][checkpoint].feature_names
            values = np.concatenate((values, shap_values_per_checkpoint.values), axis=0)
            features = np.concatenate((features, shap_values_per_checkpoint.data), axis=0)
            checkpoints = checkpoints + [(checkpoint + 1) * 10 for i in range(shap_values_per_checkpoint.shape[0])]

    checkpoints = np.array(checkpoints)

    order = np.array([i for i in range(len(feature_names))])

    # default color:
    if color is None:
        if features is not None:
            color = colors.red_blue
        else:
            color = colors.blue_rgb
    color = convert_color(color)

    idx2cat = None

    num_features = values.shape[1]

    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(num_features)])

    if log_scale:
        pl.xscale('symlog')

    feature_order = convert_ordering(order, Explanation(np.abs(values)))
    feature_inds = feature_order[:max_display]
    yticklabels = [feature_names[i] for i in feature_inds]


    row_height = 0.4
    pl.gcf().set_size_inches(plot_size[0], plot_size[1])
    pl.axvline(x=0, color="#999999", zorder=-1)

    # make the beeswarm dots
    for checkpoint in range(10, max(checkpoints), 10):
        for pos, i in enumerate(reversed(feature_inds)):
            cur_checkpoint_indices = np.where(checkpoints == checkpoint)[0]
            if len(cur_checkpoint_indices) == 0:
                continue
            pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
            shaps = values[:, i][cur_checkpoint_indices]
            fvalues = None if features is None else features[:, i][cur_checkpoint_indices]
            inds = np.arange(len(shaps))
            np.random.shuffle(inds)
            if fvalues is not None:
                fvalues = fvalues[inds]
            shaps = shaps[inds]
            colored_feature = True
            try:
                if idx2cat is not None and idx2cat[i]: # check categorical feature
                    colored_feature = False
                else:
                    fvalues = np.array(fvalues, dtype=np.float64)  # make sure this can be numeric
            except Exception:
                colored_feature = False
            N = len(shaps)
            # hspacing = (np.max(shaps) - np.min(shaps)) / 200
            # curr_bin = []
            nbins = 100
            quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
            inds = np.argsort(quant + np.random.randn(N) * 1e-6)
            layer = 0
            last_bin = -1
            ys = np.zeros(N)
            for ind in inds:
                if quant[ind] != last_bin:
                    layer = 0
                ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = quant[ind]
            ys *= 0.9 * (row_height / np.max(ys + 1))

            if safe_isinstance(color, "matplotlib.colors.Colormap") and features is not None and colored_feature:
                # TODO: this should be adjusted to not be hardcoded
                vmin = 0
                vmax = 600

                cvals = checkpoints[cur_checkpoint_indices]
                cvals_imp = cvals.copy()
                cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
                cvals[cvals_imp > vmax] = vmax
                cvals[cvals_imp < vmin] = vmin
                pl.scatter(shaps, pos + ys,
                            cmap=color, vmin=vmin, vmax=vmax, s=s,
                            c=cvals, alpha=alpha, linewidth=0,
                            zorder=3, rasterized=len(shaps) > 500)
            else:

                pl.scatter(shaps, pos + ys, s=s, alpha=alpha, linewidth=0, zorder=3,
                            color=color if colored_feature else "#777777", rasterized=len(shaps) > 500)


    # draw the color bar
    if safe_isinstance(color, "matplotlib.colors.Colormap") and color_bar and features is not None:
        import matplotlib.cm as cm
        m = cm.ScalarMappable(cmap=color)
        m.set_array([0, 1])
        cb = pl.colorbar(m, ax=pl.gca(), ticks=[0, 1], aspect=80)
        cb.set_ticklabels(["0s", "600s"])
        cb.set_label("Checkpoint", size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
#         bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
#         cb.ax.set_aspect((bbox.height - 0.9) * 20)
        # cb.draw_all()

    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    pl.yticks(range(len(feature_inds)), reversed(yticklabels), fontsize=13)
    pl.gca().tick_params('y', length=20, width=0.5, which='major')
    pl.gca().tick_params('x', labelsize=11)
    pl.ylim(-1, len(feature_inds))
    pl.xlabel(labels['VALUE'], fontsize=13)
    if show:
        pl.show()
    else:
        return pl.gca()

def shapley_boxlpot(shap_values, max_display=10, order=Explanation.abs.mean(0),
             clustering=None, cluster_threshold=0.5, color=None,
             axis_color="#333333", alpha=1, show=True, log_scale=False,
             color_bar=True, s=16, plot_size="auto", color_bar_label=labels["FEATURE_VALUE"]):
    """Create a SHAP beeswarm plot, colored by feature values when they are provided.

    Parameters
    ----------
    shap_values : Explanation
        This is an :class:`.Explanation` object containing a matrix of SHAP values
        (# samples x # features).

    max_display : int
        How many top features to include in the plot (default is 10, or 7 for
        interaction plots).

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot to be customized further
        after it has been created, returning the current axis via plt.gca().

    color_bar : bool
        Whether to draw the color bar (legend).

    s : float
        What size to make the markers. For further information see `s` in ``matplotlib.pyplot.scatter``.

    plot_size : "auto" (default), float, (float, float), or None
        What size to make the plot. By default, the size is auto-scaled based on the
        number of features that are being displayed. Passing a single float will cause
        each row to be that many inches high. Passing a pair of floats will scale the
        plot by that number of inches. If ``None`` is passed, then the size of the
        current figure will be left unchanged.

    Examples
    --------
    See `beeswarm plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html>`_.

    """
    pl.rcParams['font.family'] = 'serif'
    pl.rcParams['font.serif'] = 'Times New Roman'
    checkpoints = []
    experiments = []
    values, features, feature_names = None, None, None
    for experiment, shapleys_per_experiment in shap_values.items():
        for fold, shap_values_per_fold in enumerate(shapleys_per_experiment):
            for checkpoint, shap_values_per_checkpoint in enumerate(shap_values_per_fold):
                if len(shap_values_per_checkpoint) == 0:
                    continue
                if values is None:
                    values = np.empty((0, shap_values_per_fold[checkpoint].values.shape[1]))
                    features = np.empty((0, shap_values_per_fold[checkpoint].data.shape[1]))
                    feature_names = shap_values_per_fold[checkpoint].feature_names
                values = np.concatenate((values, shap_values_per_checkpoint.values), axis=0)
                features = np.concatenate((features, shap_values_per_checkpoint.data), axis=0)
                checkpoints = checkpoints + [(checkpoint + 1) * 10 for i in range(shap_values_per_checkpoint.shape[0])]
                experiments = experiments + [list(shap_values.keys()).index(experiment) for i in range(shap_values_per_checkpoint.shape[0])]

    checkpoints = np.array(checkpoints)
    experiments = np.array(experiments)
    feature_names_tex = []
    for feature in feature_names:
        feature_names_tex.append(FEATURES_TO_TEX[feature])
    feature_names = feature_names_tex

    feature_order = np.array([i for i in range(len(feature_names))])

    # default color:
    if color is None:
        if features is not None:
            color = colors.red_blue
        else:
            color = colors.blue_rgb
    color = convert_color(color)

    idx2cat = None

    num_features = values.shape[1]

    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(num_features)])

    if log_scale:
        pl.xscale('symlog')

    # feature_order = convert_ordering(order, Explanation(np.abs(values)))
    feature_inds = feature_order[:max_display]
    yticklabels = [feature_names[i] for i in feature_inds]


    row_height = .4
    plt.gcf().set_size_inches(plot_size)
    plt.axvline(x=0, color="#999999", zorder=-1)

    # make the beeswarm dots
    for pos, i in enumerate(reversed(feature_inds)):
        plt.boxplot(
            x=values[:, i],
            positions=[pos],
            vert=False,
            zorder=2,
            widths=[.8],
            showfliers=False,
            boxprops=dict(linewidth=4),
            whiskerprops=dict(linewidth=4),
            medianprops=dict(linewidth=0),
            capprops=dict(linewidth=4),
        )
        for experiment in shap_values:
            experiment_index = list(shap_values.keys()).index(experiment)
            experiment_indices = np.where(experiments == experiment_index)[0]
            for checkpoint in range(10, max(checkpoints), 10):
                cur_checkpoint_indices = np.where(checkpoints[experiment_indices] == checkpoint)[0]
                if len(cur_checkpoint_indices) == 0:
                    continue
                pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
                shaps = values[:, i][experiment_indices][cur_checkpoint_indices]
                fvalues = None if features is None else features[:, i][experiment_indices][cur_checkpoint_indices]
                inds = np.arange(len(shaps))
                np.random.shuffle(inds)
                if fvalues is not None:
                    fvalues = fvalues[inds]
                shaps = shaps[inds]
                colored_feature = True
                try:
                    if idx2cat is not None and idx2cat[i]: # check categorical feature
                        colored_feature = False
                    else:
                        fvalues = np.array(fvalues, dtype=np.float64)  # make sure this can be numeric
                except Exception:
                    colored_feature = False
                N = len(shaps)
                # hspacing = (np.max(shaps) - np.min(shaps)) / 200
                # curr_bin = []
                nbins = 100
                quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
                inds = np.argsort(quant + np.random.randn(N) * 1e-6)
                layer = 0
                last_bin = -1
                ys = np.zeros(N)
                for ind in inds:
                    if quant[ind] != last_bin:
                        layer = 0
                    ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                    layer += 1
                    last_bin = quant[ind]
                ys *= 0.9 * (row_height / np.max(ys + 1))

                # TODO: this should be adjusted to not be hardcoded
                vmin = 0
                vmax = 600

                cvals = checkpoints[experiment_indices][cur_checkpoint_indices]
                cvals_imp = cvals.copy()
                cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
                cvals[cvals_imp > vmax] = vmax
                cvals[cvals_imp < vmin] = vmin
                plt.scatter(shaps, pos + ys,
                            cmap=color, vmin=vmin, vmax=vmax, s=s,
                            c=cvals, alpha=.05, linewidth=0,
                            zorder=1, rasterized=len(shaps) > 500,)

    # draw the color bar
    if safe_isinstance(color, "matplotlib.colors.Colormap") and color_bar and features is not None:
        import matplotlib.cm as cm
        m = cm.ScalarMappable(cmap=color)
        m.set_array([0, 1])
        cb = pl.colorbar(m, ax=pl.gca(), ticks=[0, 1], aspect=80)
        cb.set_ticklabels(["0s", "600s"])
        cb.set_label("Checkpoint", size=30, labelpad=0)
        cb.ax.tick_params(labelsize=30, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
#         bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
#         cb.ax.set_aspect((bbox.height - 0.9) * 20)
        # cb.draw_all()

    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    pl.yticks(range(len(feature_inds)), reversed(yticklabels), fontsize=35,)
    pl.gca().tick_params('y', length=20, width=0.5, which='major')
    pl.gca().tick_params('x', labelsize=35)
    pl.ylim(-1, len(feature_inds))
    pl.xlabel(labels['VALUE'], fontsize=35)
    if show:
        pl.show()
    else:
        return pl.gca()


def aggregate_over_all_benchmarks():
    results_path = "./results/feature_ablation/shapley_continuous_classification"
    experiments = os.listdir(results_path)
    for verifier in SUPPORTED_VERIFIERS:
        # color_bar_plotted = False
        experiment_shapleys = {}
        for experiment in experiments:
            if not os.path.exists(f"{results_path}/{experiment}/{verifier}/shapley_values.pkl"):
                continue
            with open(f"{results_path}/{experiment}/{verifier}/shapley_values.pkl", "rb") as f:
                shapley_values = pickle.load(f)
            experiment_shapleys[experiment] = shapley_values
            # beeswarm_checkpoint_coloring(
            #     shapley_values,
            #     max_display=100,
            #     plot_size=(20,20),
            #     show=False,
            #     color_bar=not color_bar_plotted,
            # )
            # color_bar_plotted = True
        shapley_boxlpot(
            experiment_shapleys,
            plot_size=(20,20),
            show=False,
            max_display=1000,
            color_bar=True if verifier == OVAL else False
        )
        # plt.title(f"{VERIFIER_TO_TEX[verifier]}", fontsize=80)
        plt.tight_layout()
        plt.savefig(f"{results_path}/shapley_values_aggregated_{verifier}.png")
        plt.close()


if __name__ == "__main__":
    aggregate_over_all_benchmarks()
