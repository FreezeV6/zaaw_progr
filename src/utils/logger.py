import logging

def setup_logger(logfile="run.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        handlers=[logging.FileHandler(logfile), logging.StreamHandler()]
    )
    return logging.getLogger()