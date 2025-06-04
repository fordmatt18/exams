import logging
import sys
import json # To pretty-print the config dict
import collections.abc # To check for Mapping and Sequence

# Define the LoggerWriter class at the module level
class LoggerWriter:
    def __init__(self, logger_method, original_stream):
        """
        A file-like object that redirects writes to a logger method.

        Args:
            logger_method: The logger method to call (e.g., logging.info, logging.error).
            original_stream: The original stream (e.g., sys.stdout) to keep a reference
                             if needed for direct writes or restoring.
        """
        self.logger_method = logger_method
        self.original_stream = original_stream # Keep a reference
        self.linebuf = '' # Buffer for incomplete lines

    def write(self, buf):
        """
        Writes the buffer to the logger.
        Handles lines ending with newline characters.
        """
        # If the buffer contains newlines, process them
        if '\n' in buf:
            # Add any existing buffer to the start of the current buffer
            lines_to_log = (self.linebuf + buf).splitlines()
            # Log all complete lines
            for line in lines_to_log[:-1]: # All but the last part
                self.logger_method(line.rstrip())
            # The last part might be an incomplete line, store it in the buffer
            self.linebuf = lines_to_log[-1]
            # If the original buffer ended with a newline, the last part was complete
            if buf.endswith('\n'):
                if self.linebuf: # Log the last complete line if it exists
                    self.logger_method(self.linebuf.rstrip())
                self.linebuf = '' # Clear buffer as it was a complete line
        else:
            # No newline, just append to buffer
            self.linebuf += buf

    def flush(self):
        """
        Flushes any remaining buffer to the logger.
        Called when sys.stdout.flush() is called or at program exit.
        """
        if self.linebuf: # If there's anything in the buffer
            self.logger_method(self.linebuf.rstrip())
            self.linebuf = '' # Clear buffer
        # self.original_stream.flush() # Optionally flush the original stream too

    def isatty(self):
        """
        Mimics the isatty() method of a real file object.
        Important for some libraries that check if stdout is a terminal.
        """
        return self.original_stream.isatty() if hasattr(self.original_stream, 'isatty') else False

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

def setup_logging(log_file_name, console_level_str, file_level_str, config_params_dict=None, redirect_print=True):
    """
    Sets up logging to both console and a file.

    Args:
        log_file_name (str): Name of the log file.
        console_level_str (str): Logging level for console (e.g., "INFO").
        file_level_str (str): Logging level for file (e.g., "DEBUG").
        config_params_dict (dict, optional): Dictionary of configuration parameters
                                             to log at the beginning of the file.
        redirect_print (bool): If True, redirects sys.stdout and sys.stderr to the logger.
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

    logger.info("Logging setup complete. Output will be sent to console and %s", log_file_name)

    if config_params_dict:
        logger.info("--- Configuration Parameters Used for this Run ---")
        
        # Use the new deep serializable function
        serializable_config = _deep_make_serializable(config_params_dict)
        
        try:
            config_str = json.dumps(serializable_config, indent=4, sort_keys=True)
            for line in config_str.splitlines():
                logger.info(line)
        except TypeError as e:
            logger.error(f"Could not serialize config for logging: {e}")
            # Fallback: iterate and log, knowing some values might not be pretty
            for key, value in config_params_dict.items(): # Log original if deep serialization fails
                 logger.info(f"CONFIG - {key}: {value}")
        logger.info("--------------------------------------------------")

    if redirect_print:
        # Keep a reference to the original stdout/stderr before redirecting
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Redirect sys.stdout to logger.info
        # Pass the original_stdout to LoggerWriter if it needs to call original methods (like isatty)
        sys.stdout = LoggerWriter(logging.info, original_stdout)
        # Redirect sys.stderr to logger.error
        sys.stderr = LoggerWriter(logging.error, original_stderr)
        logging.info("sys.stdout and sys.stderr have been redirected to the logger.")
        # Test the redirection:
        # print("This print statement should now go through the logger (INFO level).")
        # sys.stderr.write("This stderr write should now go through the logger (ERROR level).\n")