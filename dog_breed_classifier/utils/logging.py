# logging_utils.py
import logging


class Logger:
    def __init__(self, name, log_file=None):
        # Initialize a logger instance with a given name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)  # Set the logging level to INFO
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        if log_file:
            # If a log file is provided, configure a file handler to log to the file
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Configure a console handler to log to the console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def info(self, message):
        # Log an INFO level message
        self.logger.info(message)

    def error(self, message):
        # Log an ERROR level message
        self.logger.error(message)

    # You can add other logging methods (e.g., warning, debug) as needed
