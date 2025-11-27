"""Miscellaneous utility functions for OASVD experiments."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import yaml


def set_seed(seed: int | None) -> np.random.Generator:
    """Set the global NumPy random seed and return a generator.

    Parameters
    ----------
    seed : int or None
        Seed for the random number generator.  If ``None``, a random seed is
        drawn from the operating system.

    Returns
    -------
    rng : numpy.random.Generator
        A NumPy random number generator initialised with the given seed.
    """
    if seed is None:
        seed = np.random.SeedSequence().entropy
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # for legacy APIs
    return rng


@contextmanager
def timer(message: str | None = None):
    """A context manager for timing a block of code.

    Parameters
    ----------
    message : str, optional
        If provided, this string is printed together with the elapsed time
        upon exit.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        elapsed = end - start
        if message:
            print(f"{message}: {elapsed:.3f} s")


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to a YAML file.

    Returns
    -------
    cfg : dict
        Configuration dictionary.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg