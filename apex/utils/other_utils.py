# Standard
import errno
import logging
import os
import os.path as osp
from typing import Any


def flatten_list(list_of_lists: list[list[Any]]) -> list[Any]:
    """Flattens a list of lists into a single list."""
    return [item for sublist in list_of_lists for item in sublist]


def int2mix(number: int, radix: list[int]) -> list[int]:
    assert isinstance(radix, list) and all(isinstance(i, int) for i in radix)
    mix = []
    radix_rev = radix[::-1]
    for i in range(0, len(radix_rev)):
        mix.append(number % radix_rev[i])
        number //= radix_rev[i]
    if number > 0:
        raise ValueError
    mix.reverse()
    return mix


def makedirs(path, mode=0o755):
    try:
        os.makedirs(
            os.path.expanduser(os.path.normpath(path)),
            mode=mode,
            exist_ok=True,
        )
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def init_logging(
    level=logging.INFO,
    fmt="[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s] %(levelname)s: %(message)s",  # noqa: E501
    local_logger=None,
    filename=None,
    force=False,
):
    """
    Initialize the default logger and quiet messages from common libraries
    """
    logging.basicConfig(level=level, format=fmt, force=force)

    logger = local_logger if local_logger else logging.getLogger()
    logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    logger.handlers = [stream_handler]

    if filename is not None:
        file_handler = logging.FileHandler(filename=filename, mode="a+")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.handlers.append(file_handler)
