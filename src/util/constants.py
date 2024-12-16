SAT = 0
UNSAT = 1
TIMEOUT = 2

ABCROWN = "ABCROWN"
VERINET = "VERINET"
OVAL = "OVAL-BAB"

ABCROWN_FEATURE_NAMES = [
    "Prediction Margin",
    "CROWN Global Bound",
    "aCROWN Global Bound",
    "PGD Margin",
    "No. Unstable Neurons",
    "No. of Domains",
    "No. of visited Domains",
    "Positive Domain Ratio",
    "BaB Lower Bound",
    "Tree Depth",
    "BaB Round",
    "Time since last batch",
    "Time taken for last batch"
]

OVAL_FEATURE_NAMES = [
    "Prediction Margin",
    "Initial Bound Min",
    # TODO: Rename to Adv. Attack Margin
    "Adv. Attack Margin",
    "Improved Bound Min",
    "Improved Bound Max",
    "No. Unstables",
    "Cur. No. of Domains",
    "No. of visited domains",
    "BaB Cur Global Lower Bound",
    "BaB Cur Global Upper Bound",
    "Tree Depth",
    "Positive Domain Ratio",
    "No. Batches",
    "Time since last Batch",
    "Time taken for last Batch"
]

VERINET_FEATURE_NAMES = [
    "Prediction Margin",
    "One Shot Global Bound Min",
    "One Shot Global Bound Max",
    "Percentage Safe Constraints",
    "Attack Margin",
    "No. of Unstables",
    "No. of Domains",
    "No. of visited Domains",
    "Positive Domain Ratio",
    "BaB Lower Bound",
    "Tree Depth",
    "Time since last report"
]

FEATURES_TO_TEX = {
    "Prediction Margin": "Prediction \n Margin",
    "CROWN Global Bound": "Initial \n Incomplete Bound",
    "aCROWN Global Bound": "Improved \n Incomplete Bound",
    "PGD Margin": "Adversarial \n Attack Margin",
    "No. Unstable Neurons": "Number of \n unstable Neurons",
    "No. of Domains": "Number of Domains",
    "No. of visited Domains": "Number of \n bounded Domains",
    "Positive Domain Ratio": "Ratio of \n verified Domains",
    "BaB Lower Bound": "Current global \n lower Bound",
    "Tree Depth": "BaB Tree Depth",
    "BaB Round": "Number of \n GPU Batches",
    "Time since last batch": "Computation Time \n of current Batch",
    "Time taken for last batch": "Computation Time \n of last Batch",
    "One Shot Global Bound Min": "Initial incomplete \n lower Bound",
    "One Shot Global Bound Max": "Initial incomplete \n upper Bound",
    "Percentage Safe Constraints": "Initial Percentage \n of safe Constraints",
    "Attack Margin": "Adversarial \n Attack Margin",
    "No. of Unstables": "Number of \n unstable Neurons",
    "Time since last report": "Time since \n last Report",
    "Initial Bound Min": "Initial incomplete \n lower Bound",
    "Adv. Attack Margin": "Adversarial \n Attack Margin",
    "Improved Bound Min": "Improved incomplete \n lower Bound",
    "Improved Bound Max": "Improved incomplete \n upper Bound",
    "No. Unstables": "Number of \n unstable Neurons",
    "Cur. No. of Domains": "Number of Domains",
    "No. of visited domains": "Number of \n bounded Domains",
    "BaB Cur Global Lower Bound": "Current global \n lower Bound",
    "BaB Cur Global Upper Bound": "Current global \n upper Bound",
    "No. Batches": "Number of \n GPU Batches",
    "Time since last Batch": "Computation Time \n of current Batch",
    "Time taken for last Batch": "Computation Time \n of last Batch",
}

VERIFIER_TO_TEX = {
    ABCROWN: r"$\alpha\beta$-CROWN",
    VERINET: r"VeriNet",
    OVAL: r"Oval",
}

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