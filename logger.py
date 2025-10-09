import logging
from logging.handlers import TimedRotatingFileHandler

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup a logger with a timed rotating file handler."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = TimedRotatingFileHandler(log_file, when='midnight', interval=1, backupCount=7)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

