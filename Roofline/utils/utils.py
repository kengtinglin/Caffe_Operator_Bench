import argparse
import json


def config_parse():
    parser = argparse.ArgumentParser(
        description="Parse Config file.")

    parser.add_argument(
        "--config-file", type=str, default=None)
    args = parser.parse_args()
    file_path = args.config_file
    with open(file_path, 'r') as f:
        config = json.load(f)

    return config
