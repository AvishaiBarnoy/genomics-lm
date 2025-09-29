import logging, sys, pathlib

def get_logger(name="run", log_file=None):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    if log_file:
        pathlib.Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

