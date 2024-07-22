import os


def load_log_file(file_path):
    if not os.path.exists(file_path):
        return ""
    with open(file_path, 'r', encoding='u8') as f:
        return f.read()
