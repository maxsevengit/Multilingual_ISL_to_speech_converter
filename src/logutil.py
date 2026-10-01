"""Shared logging setup. Scripts call configure_logging() once."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger("isl")
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"isl.{name}")
