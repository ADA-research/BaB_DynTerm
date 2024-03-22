# Running Time Prediction and Dynamic Algorithm Termination for Branch-and-Bound-based Neural Network Verification

This is the accompanying repository to the paper 
"Running Time Prediction and Dynamic Algorithm Termination for Branch-and-Bound-based Neural Network Verification".


It includes the following modules:

- `experiments`: Includes scripts and configuration files to rerun the experiments provided in the paper. 
The configuration files will be explained in more detail later.
- `src`: The actual source code used to train running time prediction models and to evaluate them in application
scenarios such as predicting running times using regression models or prematurely terminating unsolvable instances
in order to save compute resources. The `src` module contains the following submodules:
    - `eval`: Code to evaluate experiment results using several metrics. 
    - `parsers` Code to obtain feature values by parsing log files of the examined verification tools.
    - `running_time_prediction`: Code to evaluate our feature's capabilities in the context of running time 
  regression and timeout prediction (both with a fixed and continuous feature collection phase)
    - `util` Several helpful functions and constants, e.g. to load or merge log files. Additionally, it includes 
  a module that creates visualisations of obtained results, e.g. Scatter- or ECDF-Plots.

- `verification_logs` Includes the logs obtained by running the verification tools on the benchmarks described in
the paper.

-----------------------------------------------------------------------------------------------------

## Reproduction of Paper Results
Perform the following steps in a terminal. We give instructions for UNIX-based systems, but the procedure
is very similar on Windows.

1. Create `venv` by running `python3 -m venv ./venv` and activate it (`source ./venv/bin/activate`)
2. Install needed dependencies
   1. `pip3 install -r requirements.txt`
3. Run `python3 run_all_experiments.py`

After successful termination of the script, you should find three folds under `./results`
- `results_running_time_regression` corresponding to the results presented in Appendix A
- `results_timeout_classification` corresponding to the results presented in Section 5.1 - Fixed Feature Collection Phase
- `results_continuous_timeout_classification` corresponding to the results presented in Section 5.1 - Continuous Feature Collection Phase

These folders include plots (Scatter Plots, ECDF Plots, Confusion Matrices) as well as metrics per fold and average metrics in `metrics.json`
(in case of different thresholds `metrics_thresh_{thresh}.json`.
Notice, that the scatter plots provided in Figure B.1 (Appendix) are created under
`results/running_time_regression/{experiment_name}/{verifier}/scatter_plot.pdf`.

Finally, you can create the tables displayed in the paper by running `python3 create_all_tables.py`.
Notice, that the `results` folder must be filled already before creating the corresponding tables.
After the script ran successfully, you find the folder `tables` populated with csv files corresponding to the tables of
the paper.

The mapping of the created `.csv` files to the tables in the paper is as follows:

- `table_timeouts_fixed_feature_collection.csv` - Table 2
- `table_timeouts_continuous_feature_collection.csv` - Table 3
- `table_timeout_termination.csv` - Table 4

-------------------------------------------------------------------------------------------------------------

## Appendix

If you want to reproduce the paper results shown in the Appendix (i.e. the results for continuous timeout classification
and termination for theta=0.8/0.9), first run the script `run_experiments_appendix.py`. As before, the `results` folder
will then get populated with the results for the additional thresholds.
After that, run the `create_tables_appendix.py` script and the tables shown in the appendix will get created in `tables`
in csv format.

The mapping of the created `.csv` files to the tables in the Appendix is as follows:

- `table_timeouts_continuous_feature_collection_appendix.csv` - Table C.1
- `table_timeouts_termination_appendix.csv` - Table C.2


--------------------------------------------------------------------------------------------------------------
 ## Experiment Configurations

If you want to run your own experiments using the presented approach, you can add your own configuration in 
`experiments/running_time_prediction/config.py` and then run the respective experiment by calling 
`run_running_time_regression_experiments_from_config` or `run_timeout_classification_experiments_from_config` respectively.

The different possibilities for adjusting the experiments will now be explained in more detail:

### Running Time Prediction

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
    # point in time to stop feature collection
    "FEATURE_COLLECTION_CUTOFF": 20,
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
            "first_classification_at": 30,
            "no_classes": 200
        },
        "CIFAR_100": {
            "neuron_count": 55460,
            "first_classification_at": 30,
            "no_classes": 100
        },
        "VIT": {
            "neuron_count": 2760,
            "first_classification_at": 20
        }
    }
}
```

### Timeout Prediction
```python
CONFIG_CONTINUOUS_TIMEOUT_CLASSIFICATION = {
    # Path that holds logs of verification runs including features
    "VERIFICATION_LOGS_PATH": "./verification_logs/",
    # name of log files of the respective verifiers
    # the program expects a folder structure of VERIFICATION_LOGS_PATH/{experiment_name}/ABCROWN_LOG_NAME for example
    "ABCROWN_LOG_NAME": "abCROWN.log",
    "OVAL_BAB_LOG_NAME": "OVAL-BAB.log",
    "VERINET_LOG_NAME": "VERINET.log",
    # path to store results to
    "RESULTS_PATH": "./results/results_continuous_timeout_classification",
    # subfolders from VERIFICATION_LOGS_PATH to run experiments for. If empty array, run experiment for all found subfolders.
    "INCLUDED_EXPERIMENTS": [],
    # if features are collected continuously ("ADAPTIVE") or give an int to perform classification at a fixed point in time
    "FEATURE_COLLECTION_CUTOFF": "ADAPTIVE",
    # needed if "ADAPTIVE" feature collection cutoff is chosen, refers to the frequency of checkpoints at which classification is performed
    "TIMEOUT_CLASSIFICATION_FREQUENCY": 10,
    # maximum running time
    "MAX_RUNNING_TIME": 600,
    # change if predictions should be made incomplete results (i.e. instances solved during feature collection)
    "INCLUDE_INCOMPLETE_RESULTS": True,
    # confidence thresholds that must be exceeded s.t. a positive example is labeled as such
    "TIMEOUT_CLASSIFICATION_THRESHOLDS": [0.5, 0.99],
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
            "first_classification_at": 30,
            "no_classes": 200
        },
        "CIFAR_100": {
            "neuron_count": 55460,
            "first_classification_at": 30,
            "no_classes": 100
        },
        "VIT": {
            "neuron_count": 2760,
            "first_classification_at": 20
        }
    }
}
```