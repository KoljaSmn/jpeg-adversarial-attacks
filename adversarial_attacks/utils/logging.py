import inspect
import logging

from adversarial_attacks.config.config import Config

level = {'INFO': logging.INFO, 'ERROR': logging.ERROR, 'WARNING': logging.WARNING, 'DEBUG': logging.DEBUG,
         'CRITICAL': logging.CRITICAL}

logging.basicConfig(level=level[Config.LOGGING_LEVEL], filename=Config.LOG_FILE,
                    format='%(asctime)s %(name)-5s %(levelname)-8s %(message)s')


def __log(text, log_type='INFO', file=True):
    levels = {'INFO': logging.info, 'ERROR': logging.error, 'DEBUG': logging.debug, 'WARNING': logging.warning}
    frame = inspect.stack()[2]
    frameinfo = inspect.getframeinfo(frame[0])
    filename, lineno = frameinfo[0], frameinfo[1]
    if file:
        levels[log_type]('{}:{}   {}'.format(filename, lineno, text))
    else:
        levels[log_type](text)


def info(text, file=True):
    __log(text, 'INFO', file)


def warning(text, file=True):
    __log(text, 'WARNING', file)


def error(text, file=True):
    __log(text, 'ERROR', file)


def debug(text, file=True):
    __log(text, 'DEBUG', file)
