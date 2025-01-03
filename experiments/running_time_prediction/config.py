CONFIG_RUNNING_TIME_REGRESSION = {
    "VERIFICATION_LOGS_PATH": "./verification_logs/",
    "ABCROWN_LOG_NAME": "abCROWN.log",
    "OVAL_BAB_LOG_NAME": "OVAL-BAB.log",
    "VERINET_LOG_NAME": "VERINET.log",
    "RESULTS_PATH": "./results/results_running_time_regression",
    "INCLUDED_EXPERIMENTS": [],
    # If empty, use all folders in ./verification_logs
    "FEATURE_COLLECTION_CUTOFF": 10,
    "MAX_RUNNING_TIME": 600,
    "INCLUDE_TIMEOUTS": True,
    "INCLUDE_INCOMPLETE_RESULTS": True,
    "RANDOM_STATE": 42,
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

CONFIG_TIMEOUT_CLASSIFICATION = {
    "VERIFICATION_LOGS_PATH": "./verification_logs/",
    "ABCROWN_LOG_NAME": "abCROWN.log",
    "OVAL_BAB_LOG_NAME": "OVAL-BAB.log",
    "VERINET_LOG_NAME": "VERINET.log",
    "RESULTS_PATH": "./results/results_timeout_classification",
    # If empty, use all folders in ./verification_logs
    "INCLUDED_EXPERIMENTS": [],
    "FEATURE_COLLECTION_CUTOFF": 10,
    "INCLUDE_INCOMPLETE_RESULTS": True,
    "TIMEOUT_CLASSIFICATION_THRESHOLDS": [0.5, 0.9, 0.99],
    "RANDOM_STATE": 42,
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

CONFIG_DYNAMIC_ALGORITHM_TERMINATION = {
    "VERIFICATION_LOGS_PATH": "./verification_logs/",
    "ABCROWN_LOG_NAME": "abCROWN.log",
    "OVAL_BAB_LOG_NAME": "OVAL-BAB.log",
    "VERINET_LOG_NAME": "VERINET.log",
    "RESULTS_PATH": "./results/results_dynamic_algorithm_termination",
    # If empty, use all folders in ./verification_logs
    "INCLUDED_EXPERIMENTS": [],
    "FEATURE_COLLECTION_CUTOFF": "ADAPTIVE",
    "TIMEOUT_CLASSIFICATION_FREQUENCY": 10,
    "FIRST_CLASSIFICATION_AT": 10,
    "MAX_RUNNING_TIME": 600,
    "INCLUDE_INCOMPLETE_RESULTS": True,
    "TIMEOUT_CLASSIFICATION_THRESHOLDS": [0.99],
    "NUM_WORKERS": 10,
    "RANDOM_STATE": 42,
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

CONFIG_DYNAMIC_ALGORITHM_TERMINATION_APPENDIX = {
    "VERIFICATION_LOGS_PATH": "./verification_logs/",
    "ABCROWN_LOG_NAME": "abCROWN.log",
    "OVAL_BAB_LOG_NAME": "OVAL-BAB.log",
    "VERINET_LOG_NAME": "VERINET.log",
    "RESULTS_PATH": "./results/results_dynamic_algorithm_termination",
    # If empty, use all folders in ./verification_logs
    "INCLUDED_EXPERIMENTS": [],
    "FEATURE_COLLECTION_CUTOFF": "ADAPTIVE",
    "TIMEOUT_CLASSIFICATION_FREQUENCY": 10,
    "FIRST_CLASSIFICATION_AT": 10,
    "MAX_RUNNING_TIME": 600,
    "INCLUDE_INCOMPLETE_RESULTS": True,
    "TIMEOUT_CLASSIFICATION_THRESHOLDS": [0.5, 0.9],
    "NUM_WORKERS": 10,
    "RANDOM_STATE": 42,
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

CONFIG_DYNAMIC_ALGORITHM_TERMINATION_THETA_STUDY = {
    "VERIFICATION_LOGS_PATH": "./verification_logs/",
    "ABCROWN_LOG_NAME": "abCROWN.log",
    "OVAL_BAB_LOG_NAME": "OVAL-BAB.log",
    "VERINET_LOG_NAME": "VERINET.log",
    "RESULTS_PATH": "./results/results_dynamic_algorithm_termination",
    # If empty, use all folders in ./verification_logs
    "INCLUDED_EXPERIMENTS": [],
    "FEATURE_COLLECTION_CUTOFF": "ADAPTIVE",
    "TIMEOUT_CLASSIFICATION_FREQUENCY": 10,
    "FIRST_CLASSIFICATION_AT": 10,
    "MAX_RUNNING_TIME": 600,
    "INCLUDE_INCOMPLETE_RESULTS": True,
    "TIMEOUT_CLASSIFICATION_THRESHOLDS": [theta / 100 for theta in range(50, 100, 1)],
    "NUM_WORKERS": 10,
    "RANDOM_STATE": 42,
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

CONFIG_TIMEOUT_BASELINE = {
    "VERIFICATION_LOGS_PATH": "./verification_logs/",
    "ABCROWN_LOG_NAME": "abCROWN.log",
    "OVAL_BAB_LOG_NAME": "OVAL-BAB.log",
    "VERINET_LOG_NAME": "VERINET.log",
    "RESULTS_PATH": "./results/results_baseline_timeout_classification",
    # If empty, use all folders in ./verification_logs
    "INCLUDED_EXPERIMENTS": [],
    "TIMEOUT_CLASSIFICATION_FREQUENCY": 10,
    "MAX_RUNNING_TIME": 600,
    "INCLUDE_INCOMPLETE_RESULTS": True,
    "RANDOM_STATE": 42,
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
            "first_classification_at": 10,
            "no_classes": 200
        },
        "CIFAR_100": {
            "neuron_count": 55460,
            "first_classification_at": 10,
            "no_classes": 100
        },
        "VIT": {
            "neuron_count": 2760,
            "first_classification_at": 10
        }
    }
}
