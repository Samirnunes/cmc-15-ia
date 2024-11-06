import sys
from logging import INFO, StreamHandler, getLogger

logger = getLogger("lab5")
logger.setLevel(INFO)
logger.addHandler(StreamHandler(sys.stdout))
