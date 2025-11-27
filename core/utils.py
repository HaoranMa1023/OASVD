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

def measure_runtime(func, *args, repeat=5, **kwargs):
    """Measure average runtime of a function over multiple repeats."""
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)
    return {
        "avg": float(np.mean(times)),
        "std": float(np.std(times)),
        "all": times,
    }

def detect_change(stat: np.ndarray,
                  threshold: float,
                  true_changes: list[int],
                  patience: int = 1):
    """
    Detect when a statistic exceeds threshold near known change points.

    Parameters
    ----------
    stat : ndarray
        Statistic time series (e.g. sigma_(r+1)).
    threshold : float
        Decision threshold.
    true_changes : list[int]
        True change point indices.
    patience : int
        Require threshold exceedance for `patience` consecutive steps.

    Returns
    -------
    delays : list[int]
        Detection delay relative to true change times.
    false_alarms : int
        Number of times threshold is crossed outside allowed windows.
    """
    T = len(stat)
    detected = [False] * len(true_changes)
    delays = []
    false_alarms = 0

    for t in range(T):
        if stat[t] >= threshold:
            # Check if near a true change
            matched = False
            for i, tc in enumerate(true_changes):
                if not detected[i] and t >= tc:
                    detected[i] = True
                    delays.append(t - tc)
                    matched = True
                    break
            if not matched:
                false_alarms += 1

    return delays, false_alarms

def save_results(path: str, results: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(results, f)
