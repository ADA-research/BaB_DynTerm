import os.path
from collections import defaultdict

import glob

from src.parsers.parse_ab_crown_log import parse_abcrown_log
from src.parsers.parse_oval_log import parse_oval_log
from src.parsers.parse_verinet_log import parse_verinet_log
from src.util.io import load_log_file
from src.util.constants import result_to_enum, SAT, TIMEOUT, UNSAT, ABCROWN, VERINET, OVAL, SUPPORTED_VERIFIERS


def fix_image_ids_in_logs_oval(log_string):
    """
    Function to replace ids in Oval verification log with continuous ids. Helpful
     if  multiple logs are merged into one because of failed/distributed runs!
    :param log_string: path to log string
    :return: fixed log file
    """
    cur_index = 0
    fixed_string = ""
    for line in log_string.splitlines():
        if "Verifying Image" in line:
            fixed_string += f"############################ Verifying Image {cur_index} ####################################\n"
            cur_index += 1
        else:
            fixed_string += line + '\n'

    return fixed_string


def fix_image_ids_in_all_oval_logs():
    for log_file in glob.glob("./verification_logs/*/OVAL-BAB.log"):
        print(log_file)
        log_string = load_log_file(log_file)
        fixed_string = fix_image_ids_in_logs_oval(log_string)
        with open(log_file, "w", encoding='u8') as f:
            f.write(fixed_string)


def fix_image_ids_in_logs_verinet(log_string):
    """
    Function to replace ids in VeriNet verification log with continuous ids. Helpful
     if  multiple logs are merged into one because of failed/distributed runs!
    :param log_string: path to log string
    :return: fixed log file
    """
    cur_index = 0
    fixed_string = ""
    lines = log_string.splitlines()
    no_lines = len(lines)
    for index, line in enumerate(lines):
        print(f"Line {index} / {no_lines}")
        if "Verifying image" in line:
            fixed_string += f"#################################################### Verifying image {cur_index} #######################################################\n"
            cur_index += 1
        else:
            fixed_string += line + '\n'

    return fixed_string


def sanity_check():
    """
    Function to perform a sanity check across different verifiers, i.e. checking if they all come to the same verification
    solution per instance
    :param experiment_dict:
        Example Dict:
        {
            "MNIST_CONV_BIG": {
                "ABCROWN": "./verification_logs/abCROWN/MNIST_CONV_BIG/ABCROWN-MNIST-CONV-BIG-39418085.log",
                "VERINET": "./verification_logs/VeriNet/MNIST_CONV_BIG/VERINET-MNIST-CONV-BIG-40450222.log",
                "OVAL": "./verification_logs/oval-bab/MNIST_CONV_BIG/MNIST_CONV_BIG_FIXED.log"
            }
        }
    """
    for experiment_name in os.listdir("./verification_logs/"):
        print(f"EXPERIMENT {experiment_name}")
        return_dict = defaultdict(dict)
        for verifier in SUPPORTED_VERIFIERS:
            if verifier == ABCROWN:
                running_time_dict = parse_abcrown_log(load_log_file(f"./verification_logs/{experiment_name}/abCROWN.log"))
            elif verifier == VERINET:
                running_time_dict = parse_verinet_log(load_log_file(f"./verification_logs/{experiment_name}/VERINET.log"))
            elif verifier == OVAL:
                running_time_dict = parse_oval_log(load_log_file(f"./verification_logs/{experiment_name}/OVAL-BAB.log"))
            else:
                assert False, "Unsupported Verifier!"

            for index, result in running_time_dict.items():
                return_dict[index][verifier] = result["result"]

        print("------------------------- DOING SANITY CHECK --------------------------------")
        for index in return_dict:
            result_set = set([result_to_enum[result] for verifier, result in return_dict[index].items()])
            print(f"Index {index}: Result {result_set}")
            if len(result_set) == 2:
                # case where UNSAT/SAT and timeout
                if {SAT, TIMEOUT} == result_set or {UNSAT, TIMEOUT} == result_set:
                    # this is ok!
                    pass
                # case where skipped and unsafe!
                elif {SAT, 3} == result_set:
                    # this is ok as alpha beta crown treats misclassified instances as SAT
                    pass
                else:
                    print(f"ERROR!!! at index {index}: {return_dict[index]}")


if __name__ == "__main__":
    sanity_check()
    #
    # verinet_log_file_path = "./bab_features/verification_logs/CIFAR_RESNET_2B/VERINET.log"
    # log_string = load_log_file(verinet_log_file_path)
    # fixed_string = fix_image_ids_in_logs_verinet(log_string)
    # with open(verinet_log_file_path, "w", encoding='u8') as f:
    #     f.write(fixed_string)