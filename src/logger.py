import logging

class Logger:
    def __init__(self, log_path):
        logging.basicConfig(filename=log_path, level=logging.INFO)
        self.logger = logging.getLogger()

    def log(self, message):
        self.logger.info(message)
