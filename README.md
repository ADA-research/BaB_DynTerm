# Running Time Prediction and Dynamic Algorithm Termination for Branch-and-Bound-based Neural Network Verification

This is the accompanying repository to the paper:

**Dynamic Algorithm Termination for Branch-and-Bound-based Neural Network Verification**, Konstantin Kaulen, Matthias König,
Holger H. Hoos.
To appear in *Proceedings of the 39th AAAI Conference on Artificial Intelligence (AAAI-25)*.

**Abstract**: With the rising use of neural networks across various 
application domains, it becomes increasingly important to ensure
that they do not exhibit dangerous or undesired behaviour. In
light of this, several neural network robustness verification 
algorithms have been developed, among which methods based
on Branch and Bound (BaB) constitute the current state of the
art. However, these algorithms still require immense 
computational resources. In this work, we seek to reduce this cost
by leveraging running time prediction techniques, thereby 
allowing for more efficient resource allocation and use.
Towards this end, we present a novel method that 
dynamically predicts whether a verification instance can be solved
in the remaining time budget available to the verification algorithm. 
We introduce features describing BaB-based verification 
instances and use these to construct running time,
and more specifically, timeout prediction models. We 
leverage these models to terminate runs on instances early in the
verification process that would otherwise result in a timeout.
Overall, using our method, we were able to reduce the 
total running time by 64% on average compared to the standard
verification procedure, while certifying a comparable number
of instances.

### Citation
If you want to cite this work, we kindly ask you to do so using the following BibTex entry:
```
@inproceedings{KauEtAl25,
    author = {Kaulen, Konstantin and K{\"o}nig, Matthias and Hoos, Holger H},
    title = "Dynamic Algorithm Termination for Branch-and-Bound-based Neural Network Verification",
    booktitle = "To appear in Proceedings of the 39th AAAI Conference on Artificial Intelligence (AAAI-25)",
    year = "2025",
    pages = "1--9"
}
```

## Structure

The repository includes the following modules:

- `experiments`: Includes scripts and configuration files to rerun the experiments provided in the paper. 
The configuration files will be explained in more detail later.
- `src`: The actual source code used to train running time prediction models and to evaluate them in application
scenarios such as predicting running times using regression models or prematurely terminating unsolvable instances.
The `src` module contains the following submodules:
    - `eval`: Code to evaluate experiment results using several metrics. 
    - `parsers` Code to obtain feature values by parsing log files of the examined verification tools.
    - `running_time_prediction`: Code to evaluate our feature's capabilities in the context of running time 
  regression and timeout prediction (both with a fixed and continuous feature collection phase)
    - `util` Several helpful functions and constants, e.g. to load or merge log files. Additionally, it includes 
  a module that creates visualisations of obtained results, e.g. Scatter- or ECDF-Plots.

-----------------------------------------------------------------------------------------------------

## Getting Started
Perform the following steps in a terminal. We give instructions for UNIX-based systems, but the procedure
is very similar on Windows. All steps assume a present installation of Python 3.11.

First, setup the dependencies:
1. Create `venv` by running `python3 -m venv ./venv` and activate it (`source ./venv/bin/activate`)
2. Install needed dependencies
   1. `pip3 install -r requirements.txt`

Now, you can run the experiments from the paper and the appendix (see below) based on pre-parsed features. If you want
to parse the features based on the original log files, please proceed as follows.

Download the log files obtained by running the verification systems on the respective benchmarks.
We use those files to extract the instance features.

```bash
wget https://rwth-aachen.sciebo.de/s/a6c4VlYRRgQE4st/download -O verification_logs.zip
unzip verification_logs.zip
```

**ATTENTION**: The current implementation does only check for the file name requested and returns the saved 
features when the log file does not exist. Once a log file is found, the current saved features get overwritten, which
is important to keep in mind when changing out log files.

## Reproduction of Results

### Main Paper

You can reproduce all experiment results presented in the main paper by executing `python3 run_experiments_main_paper.py`.

After successful termination of the script, you should find the folder `results_continuous_timeout_classification` under `./results`

These folders include plots (Scatter Plots, ECDF Plots, Confusion Matrices) as well as metrics per fold and average metrics in `metrics.json`
(in case of different thresholds `metrics_thresh_{thresh}.json`.

Finally, you can create the tables displayed in the paper by running `python3 create_tables_main_paper.py`.
Notice, that the `results` folder must be filled already before creating the corresponding tables.
After the script ran successfully, you find the folder `tables` populated with csv files corresponding to the tables of
the paper.

The mapping of the created `.csv` files to the tables in the paper is as follows:

- `benchmark_overview.csv` - Table 1
- `table_timeouts_continuous_feature_collection_0.99.csv` - Table 2
- `table_timeout_termination_0.99.csv` - Table 3


### Appendix
Please run the running time regression experiments, the experiment for different choices of $\theta$
and the Shapley Value study by executing `python3 run_experiments_appendix.py`. 

The plots corresponding to Figures 2,3 and 4 can be found under `results/shapley_value_study/shapley_values_aggregated_{verifier}.png`.
The plots in Figure 5 are saved under `results/results_running_time_regression/{BENCHMARK}/{VERIFIER}/scatter_plot.pdf`.

Then execute `create_tables_appendix.py` to create the tables in the Appendix:
- `table_timeout_termination_{0.5,0.9}.csv` - Appendix, Table 2
- `table_running_time_regression.csv` - Appendix, Table 3

Finally, to create the plots in Figure 1 of the Appendix, please execute `run_theta_study.py`. The resulting plots
can then be found in `results/resu.ts/continuous_timeout_classification/theta_distribution_{VERIFIER}.pdf`. Be aware,
that this script runs the experiments for every $\theta$ between 0.5 and 1 with a step size of 0.01, so it is quite
resource intensive.

### Feature Ablation
Please run `run_feature_ablation.py` to conduct the feature ablation study. After that,
you will find the results of the study in `./results/feature_ablation/feature_ablation_continuous_classification`.
There are separate tables per verifier and benchmark, showing the effects the exclusion of one feature has in comparison
to the baseline, where all features are used to make predictions. Those tables are named `feature_ablation_{verifier}_{benchmark}.csv`.
In addition, the tables named `feature_ablation_average_{verifier}.csv` hold the results aggregated as averages over all 
benchmarks for each verifier. Notice though, that this aggregation is not ultimately conclusive, since
several features are more important on some benchmarks than on others. This nuance often get lost when looking 
at the average importance.


--------------------------------------------------------------------------------------------------------------
 ## Experiment Configurations

If you want to run your own experiments using the presented approach, you can add your own configuration in 
`experiments/running_time_prediction/config.py` and then run the respective experiment by calling 
`run_running_time_regression_experiments_from_config` or `run_timeout_classification_experiments_from_config` respectively.
The different possibilities for adjusting the experiments will now be explained in more detail:

### Dynamic Algorithm Termination
```python
CONFIG_DYNAMIC_ALGORITHM_TERMINATION = {
    # Path that holds logs of verification runs including features
    "VERIFICATION_LOGS_PATH": "./verification_logs/",
    # name of log files of the respective verifiers
    # the program expects a folder structure of VERIFICATION_LOGS_PATH/{experiment_name}/ABCROWN_LOG_NAME for example
    "ABCROWN_LOG_NAME": "abCROWN.log",
    "OVAL_BAB_LOG_NAME": "OVAL-BAB.log",
    "VERINET_LOG_NAME": "VERINET.log",
    # path to store results to
    "RESULTS_PATH": "./results/results_dynamic_algorithm_termination",
    # subfolders from VERIFICATION_LOGS_PATH to run experiments for. If empty array, run experiment for all found subfolders.
    "INCLUDED_EXPERIMENTS": [],
    # if the predictions should be done dynamically ("ADAPTIVE") or give an int to perform classification only once 
    # at a fixed point in time
    # in the paper, we only present results for the dynamic termination
    "FEATURE_COLLECTION_CUTOFF": "ADAPTIVE",
    # parameter t_{freq} of the main paper; refers to the frequency of checkpoints at which classification is performed
    "TIMEOUT_CLASSIFICATION_FREQUENCY": 10,
    # parameter t_cutoff in the paper; maximum running time
    "MAX_RUNNING_TIME": 600,
    # change if predictions should be made incomplete results (i.e. instances solved during feature collection)
    "INCLUDE_INCOMPLETE_RESULTS": True,
    # parameter \theta of the main paper; confidence thresholds that must be exceeded s.t. a positive example is labeled as such
    "TIMEOUT_CLASSIFICATION_THRESHOLDS": [0.5, 0.99],
    # Number of processes to spawn for running the experiment
    "NUM_WORKERS": 10,
    # fixed random state in fold creation and for random forest. This leads to reproducibility of results, change if you like to!
    "RANDOM_STATE": 42,
    # Additional Info for each experiment, i.e. neuron count, no_classes and adjusted feature_collection cutoffs (first_classification_at).
    # the default value for no. classes is 10 and the default value for first_classification_at is FEATURE_COLLECTION_CUTOFF (see above)
    "EXPERIMENTS_INFO": {
        "MNIST_6_100": {
            "neuron_count": 510,
        },
        "MNIST_9_100": {
            "neuron_count": 810
        },
        "MNIST_CONV_BIG": {
            "neuron_count": 48064
        },
        "MNIST_CONV_SMALL": {
            "neuron_count": 3604
        },
        "CIFAR_RESNET_2B": {
            "neuron_count": 6244
        },
        "OVAL21": {
            "neuron_count": 6244
        },
        "MARABOU": {
            "neuron_count": 2568
        },
        "SRI_RESNET_A": {
            "neuron_count": 9316
        },
        "TINY_IMAGENET": {
            "neuron_count": 172296,
            "no_classes": 200
        },
        "CIFAR_100": {
            "neuron_count": 55460,
            "no_classes": 100
        },
        "VIT": {
            "neuron_count": 2760,
        }
    }
}
```

### Running Time Regression

```python
CONFIG_RUNNING_TIME_REGRESSION = {
    # Path that holds logs of verification runs including features
    "VERIFICATION_LOGS_PATH": "./verification_logs/",
    # name of log files of the respective verifiers
    # the program expects a folder structure of VERIFICATION_LOGS_PATH/{experiment_name}/ABCROWN_LOG_NAME for example
    "ABCROWN_LOG_NAME": "abCROWN.log",
    "OVAL_BAB_LOG_NAME": "OVAL-BAB.log",
    "VERINET_LOG_NAME": "VERINET.log",
    # path to store results to
    "RESULTS_PATH": "./results/results_running_time_regression",
    # subfolders from VERIFICATION_LOGS_PATH to run experiments for. If empty array, run experiment for all found subfolders.
    "INCLUDED_EXPERIMENTS": [],
    # point in time to stop feature collection to predict running times
    "FEATURE_COLLECTION_CUTOFF": 10,
    # maximum running time
    "MAX_RUNNING_TIME": 600,
    # change if predictions should be made for timeouts and incomplete results (i.e. instances solved during feature collection)
    "INCLUDE_TIMEOUTS": True,
    "INCLUDE_INCOMPLETE_RESULTS": True,
    # fixed random state in fold creation and for random forsest. This leads to reproducibility of results, change if you like to!
    "RANDOM_STATE": 42,
    # Additional Info for each experiment, i.e. neuron count, no_classes and adjusted feature_collection cutoffs (first_classification_at).
    # the default value for no. classes is 10 and the default value for first_classification_at is FEATURE_COLLECTION_CUTOFF (see above)
    "EXPERIMENTS_INFO": {
        "MNIST_6_100": {
            "neuron_count": 510,
        },
        "MNIST_9_100": {
            "neuron_count": 810
        },
        "MNIST_CONV_BIG": {
            "neuron_count": 48064
        },
        "MNIST_CONV_SMALL": {
            "neuron_count": 3604
        },
        "CIFAR_RESNET_2B": {
            "neuron_count": 6244
        },
        "OVAL21": {
            "neuron_count": 6244
        },
        "MARABOU": {
            "neuron_count": 2568
        },
        "SRI_RESNET_A": {
            "neuron_count": 9316
        },
        "TINY_IMAGENET": {
            "neuron_count": 172296,
            "no_classes": 200
        },
        "CIFAR_100": {
            "neuron_count": 55460,
            "no_classes": 100
        },
        "VIT": {
            "neuron_count": 2760,
        }
    }
}
```