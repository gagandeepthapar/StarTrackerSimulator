"""
CONSTANTS 

List of constants to aid in development and maintain consistency

startrackermodel
"""
import logging
import logging.config
from numpy import pi as PI
import os

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
            "datefmt": "%d/%b/%Y %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            # "stream": "ext://sys.stdout",  # Default is stderr
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["default"],
            "level": "WARNING",
            "propagate": False,
        },
        "my.packg": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "__main__": {  # if __name__ == '__main__'
            "handlers": ["default"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


# Set project level defaults
__CONST_ROOT = os.path.dirname(os.path.realpath(__file__))
SAVEDATA = os.path.join(__CONST_ROOT, "pklfiles/")
MEDIA = os.path.join(__CONST_ROOT, "..", "media/")

# Useful files
YBSC_PKL = os.path.join(__CONST_ROOT, "YBSC.pkl")

# Constants
RAD2DEG = 180.0 / PI
DEG2RAD = PI / 180.0
RAD2ARCSEC = 180.0 / PI * 3600
BOLTZMANN = 8.61733e-5  # eV/K
ELECTRON_CHARGE = 1.602e-19  # C
MU_EARTH = 398600  # km3/s2
RAD_EARTH = 6378  # km
C_TO_K = 273.15
SPEED_OF_LIGHT = 299_792_458

# Log levels
level_hash = {
    "DEBUG": logging.DEBUG,
    "D": logging.DEBUG,
    "INFO": logging.INFO,
    "I": logging.INFO,
    "WARNING": logging.WARNING,
    "W": logging.WARNING,
    "ERROR": logging.ERROR,
    "E": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "C": logging.CRITICAL,
}

if __name__ == "__main__":
    logger.info("DEBUG TEST")
    logger.info(__CONST_ROOT)
