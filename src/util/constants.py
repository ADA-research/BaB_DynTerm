SAT = 0
UNSAT = 1
TIMEOUT = 2

ABCROWN = "ABCROWN"
VERINET = "VERINET"
OVAL = "OVAL-BAB"

SUPPORTED_VERIFIERS = [
    ABCROWN,
    VERINET,
    OVAL
]

result_to_enum_abcrown = {
    "safe-incomplete": UNSAT,
    "safe": UNSAT,
    "safe-incomplete-refine": UNSAT,
    "unsafe-pgd": SAT,
    "unknown": TIMEOUT,
}

result_to_enum_verinet = {
    "Status.Safe": UNSAT,
    "Status.Unsafe": SAT,
    "Status.Undecided": TIMEOUT,
    "Status.Skipped": SAT

}

result_to_enum_oval = {
    "UNSAT": UNSAT,
    "SAT": SAT,
    "Timeout": TIMEOUT,
    "SKIPPED": SAT
}

result_to_enum = {
    **result_to_enum_abcrown,
    **result_to_enum_verinet,
    **result_to_enum_oval
}

MNIST_ERAN_EXPERIMENTS = ["MNIST_6_100", "MNIST_9_100", "MNIST_CONV_BIG", "MNIST_CONV_SMALL"]
CIFAR_EXPERIMENTS = ["CIFAR_RESNET_2B", "MARABOU", "OVAL21", "VIT", "SRI_RESNET_A"]
LARGE_RESNET_EXPERIMENTS = ["CIFAR_100", "TINY_IMAGENET"]

experiment_groups = {
    "MNIST ERAN": MNIST_ERAN_EXPERIMENTS,
    "CIFAR 10": CIFAR_EXPERIMENTS,
    "LARGE RESNET": LARGE_RESNET_EXPERIMENTS
}

experiment_samples = {
    "MNIST_6_100": 960,
    "MNIST_9_100": 947,
    "MNIST_CONV_BIG": 929,
    "MNIST_CONV_SMALL": 980,
    "CIFAR_RESNET_2B": 703,
    "MARABOU": 500,
    "OVAL21": 500,
    "VIT": 500,
    "SRI_RESNET_A": 500,
    "CIFAR_100": 500,
    "TINY_IMAGENET": 500
}