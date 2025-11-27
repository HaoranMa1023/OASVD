"""Plotting utilities for OASVD experiments."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_error_rank(time_steps: np.ndarray,
                    error_series: list[np.ndarray],
                    rank_series: list[np.ndarray],
                    labels: list[str],
                    title: str = "Error and Rank over Time",
                    ylabel_error: str = "Relative Error",
                    ylabel_rank: str = "Rank",
                    outfile: str | None = None) -> None:
    """Plot relative error and rank trajectories on a dual y‑axis.

    Parameters
    ----------
    time_steps : ndarray of shape (T,)
        Array of time indices or snapshot numbers.
    error_series : list of ndarrays
        List where each element is an array of relative errors over time for
        one method.
    rank_series : list of ndarrays
        List where each element is an array of truncation ranks over time for
        one method.  Must have the same length as ``error_series``.
    labels : list of str
        Labels corresponding to each method in the plots.
    title : str, optional
        Title of the plot.
    ylabel_error : str, optional
        Label for the left y‑axis (error).
    ylabel_rank : str, optional
        Label for the right y‑axis (rank).
    outfile : str, optional
        If provided, the figure is saved to this path instead of shown.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    # Plot error on left axis (log scale)
    for err, label in zip(error_series, labels):
        ax1.semilogy(time_steps, err, label=label)
    ax1.set_xlabel("Time step")
    ax1.set_ylabel(ylabel_error)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot rank on right axis
    ax2 = ax1.twinx()
    for ranks, label in zip(rank_series, labels):
        ax2.plot(time_steps, ranks, linestyle=':', linewidth=1.0, label=label + " (rank)")
    ax2.set_ylabel(ylabel_rank)

    # Create combined legend
    lines, labels_combined = [], []
    for ax in [ax1, ax2]:
        line, label = ax.get_legend_handles_labels()
        lines += line
        labels_combined += label
    ax1.legend(lines, labels_combined, loc='best')

    plt.title(title)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
        plt.close(fig)
    else:
        plt.show()


def plot_pareto(x: np.ndarray,
                y: np.ndarray,
                labels: list[str],
                xlabel: str,
                ylabel: str,
                title: str,
                outfile: str | None = None) -> None:
    """Plot a Pareto scatter diagram (e.g. runtime vs error).

    Parameters
    ----------
    x, y : ndarray of shape (k,)
        Coordinates of the points (e.g. runtime on x, error on y).
    labels : list of str
        Labels for each point.
    xlabel, ylabel : str
        Axis labels.
    title : str
        Title of the plot.
    outfile : str, optional
        If provided, save the figure instead of displaying it.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (x[i], y[i]), textcoords="offset points", xytext=(5, 5), ha='left')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
        plt.close(fig)
    else:
        plt.show()