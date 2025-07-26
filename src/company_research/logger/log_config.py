# log_config.py
import logging.config
import os

import yaml

parent_path = os.path.dirname(os.path.abspath(__file__))
default_config_file_path = os.path.join(parent_path, 'logging_config.yaml')

def setup_logging(config_path=default_config_file_path, default_level=logging.INFO):
    """
    Sets up logging using a YAML config file.

    Parameters:
        config_path (str): Path to the YAML logging config file.
        default_level (int): Fallback logging level if config file not found.
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
        logging.warning(f"Logging config file '{config_path}' not found. Using basic config.")
