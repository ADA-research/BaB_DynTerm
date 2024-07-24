SAT = 0
UNSAT = 1
TIMEOUT = 2

ABCROWN = "ABCROWN"
VERINET = "VERINET"
OVAL = "OVAL-BAB"

ABCROWN_FEATURE_NAMES = [
    "PGD Margin",
    "CROWN Global Bound",
    "aCROWN Global Bound",
    "No. Unstable Neurons",
    "Prediction Margin",
    "Positive Domain Ratio",
    "No. of Domains",
    "No. of visited Domains",
    "BaB Lower Bound",
    "Tree Depth",
    "BaB Round",
    "Time since last batch",
    "Time taken for last batch"
]

OVAL_FEATURE_NAMES = [
    "No. Batches",
    "Time since last Batch",
    "Prediction Margin",
    "Initial Bound Min",
    "Initial Bound Max",
    "Improved Bound Min",
    "Improved Bound Max",
    "No. Unstables",
    "BaB Cur Global Lower Bound",
    "BaB Cur Global Upper Bound",
    "No. of visited domains",
    "Cur. No. of Domains",
    "Cur No. of Hard Domains",
    "Tree Depth",
    "Time needed for Split",
    "Time needed for Branching"
]

VERINET_FEATURE_NAMES = [
    "Attack Margin",
    "One Shot Global Bound Min",
    "One Shot Global Bound Max",
    "Percentage Safe Constraints",
    "No. of Unstables",
    "Prediction Margin",
    "Positive Domain Ratio",
    "No. of Domains",
    "No. of visited Domains",
    "BaB Lower Bound",
    "Tree Depth",
    "Time since last report"
]

VERIFIER_FEATURE_MAP = {
    ABCROWN: ABCROWN_FEATURE_NAMES,
    OVAL: OVAL_FEATURE_NAMES,
    VERINET: VERINET_FEATURE_NAMES
}

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