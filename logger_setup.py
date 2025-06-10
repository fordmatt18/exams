import logging
import sys
import json # To pretty-print the config dict
import collections.abc # To check for Mapping and Sequence

def _deep_make_serializable(obj):
    """
    Recursively converts sets/frozensets in an object to sorted lists
    to make it JSON serializable.
    """
    if isinstance(obj, (set, frozenset)):
        return sorted(list(obj))
    elif isinstance(obj, collections.abc.Mapping): # For dict-like objects
        return {k: _deep_make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, collections.abc.Sequence) and not isinstance(obj, (str, bytes)): # For list/tuple-like, but not strings
        return [_deep_make_serializable(item) for item in obj]
    return obj # Return as is if not a special type to convert

def setup_logging(log_file_name, console_level_str, file_level_str, config_params_dict=None):
    """
    Sets up logging to both console and a file.

    Args:
        log_file_name (str): Name of the log file.
        console_level_str (str): Logging level for console (e.g., "INFO").
        file_level_str (str): Logging level for file (e.g., "DEBUG").
        config_params_dict (dict, optional): Dictionary of configuration parameters
                                             to log at the beginning of the file.
    """
    # Get numeric log levels
    console_level = getattr(logging, console_level_str.upper(), logging.INFO)
    file_level = getattr(logging, file_level_str.upper(), logging.DEBUG)

    # Create a logger
    logger = logging.getLogger() # Get the root logger
    logger.setLevel(min(console_level, file_level))

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Create file handler
    file_handler = logging.FileHandler(log_file_name, mode='w')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout) # Still use original sys.stdout for this handler
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logging.info("Logging setup complete. Output will be sent to console and %s", log_file_name)

    if config_params_dict:
        logging.info("--- Configuration Parameters Used for this Run ---")
        
        # Use the new deep serializable function
        serializable_config = _deep_make_serializable(config_params_dict)
        
        try:
            config_str = json.dumps(serializable_config, indent=4, sort_keys=True)
            for line in config_str.splitlines():
                logging.info(line)
        except TypeError as e:
            logging.error(f"Could not serialize config for logging: {e}")
            # Fallback: iterate and log, knowing some values might not be pretty
            for key, value in config_params_dict.items(): # Log original if deep serialization fails
                 logging.info(f"CONFIG - {key}: {value}")
        logging.info("--------------------------------------------------")