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
- `results_running_time_regression`
- `results_timeout_classification`



