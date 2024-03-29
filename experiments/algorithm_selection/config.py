CONFIG_ADAPTIVE_ALGORITHM_SELECTION = {
    "VERIFICATION_LOGS_PATH": "./verification_logs/",
    "ABCROWN_LOG_NAME": "abCROWN.log",
    "OVAL_BAB_LOG_NAME": "OVAL-BAB.log",
    "VERINET_LOG_NAME": "VERINET.log",
    "RESULTS_PATH": "./results/results_dynamic_algorithm_selection",
    # If empty, use all folders in ./verification_logs
    "INCLUDED_EXPERIMENTS": [],
    "ALGORITHM_SELECTION_FREQUENCY": 10,
    "MAX_RUNNING_TIME": 600,
    # ATTENTION: this is an EXCLUSIVE lower bound!
    "SELECTION_THRESHOLDS": [0.5, 0.99],
    "STOP_PREDICTED_TIMEOUTS": False,
    "RANDOM_STATE": 42,
    "EXPERIMENTS_INFO": {
        "MNIST_6_100": {
            "neuron_count": 510,
            # default is no_classes=10 for CIFAR-10 and MNIST
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

CONFIG_ADAPTIVE_ALGORITHM_SELECTION_AND_TERMINATION = {
    "VERIFICATION_LOGS_PATH": "./verification_logs/",
    "ABCROWN_LOG_NAME": "abCROWN.log",
    "OVAL_BAB_LOG_NAME": "OVAL-BAB.log",
    "VERINET_LOG_NAME": "VERINET.log",
    "RESULTS_PATH": "./results/results_adaptive_algorithm_selection_with_termination",
    # If empty, use all folders in ./verification_logs
    "INCLUDED_EXPERIMENTS": [],
    "ALGORITHM_SELECTION_FREQUENCY": 10,
    "MAX_RUNNING_TIME": 600,
    # ATTENTION: this is an EXCLUSIVE lower bound!
    "SELECTION_THRESHOLDS": [0.5, 0.99],
    "STOP_PREDICTED_TIMEOUTS": True,
    "RANDOM_STATE": 42,
    "EXPERIMENTS_INFO": {
        "MNIST_6_100": {
            "neuron_count": 510,
            # default is no_classes=10 for CIFAR-10 and MNIST
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
