from logging import getLogger, StreamHandler, INFO
import sys

logger = getLogger("lab5")
logger.setLevel(INFO)
logger.addHandler(StreamHandler(sys.stdout))