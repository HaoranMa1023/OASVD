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

def plot_time_series(time_steps: np.ndarray,
                     series_list: list[np.ndarray],
                     labels: list[str],
                     title: str,
                     ylabel: str,
                     logy: bool = False,
                     vlines: list[int] | None = None,
                     vline_style: dict | None = None,
                     outfile: str | None = None) -> None:
    """Plot one or more time series on a single axis.

    This is a generic helper for:
    - detection statistics vs time
    - orthogonality error gamma_t vs time
    - single rank / energy curves, etc.

    Parameters
    ----------
    time_steps : ndarray of shape (T,)
        Time indices.
    series_list : list of ndarrays
        Each element is an array of length T.
    labels : list of str
        Legend labels for each series.
    title : str
        Plot title.
    ylabel : str
        Label for y-axis.
    logy : bool, optional
        If True, use a logarithmic y-scale.
    vlines : list of int, optional
        Time indices at which to draw vertical reference lines
        (e.g. true change-points, physical events).
    vline_style : dict, optional
        Matplotlib style kwargs for vlines (color, linestyle, etc.).
    outfile : str, optional
        If given, save to this path instead of showing.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    for series, label in zip(series_list, labels):
        ax.plot(time_steps, series, label=label)

    if logy:
        ax.set_yscale("log")

    ax.set_xlabel("Time step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5)

    if vlines:
        style = {"color": "k", "linestyle": "--", "linewidth": 0.8}
        if vline_style is not None:
            style.update(vline_style)
        for t in vlines:
            ax.axvline(x=t, **style)

    if labels:
        ax.legend(loc="best")

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
        plt.close(fig)
    else:
        plt.show()


def plot_bar(categories: list[str],
             values: list[float],
             title: str,
             ylabel: str,
             rotation: int = 0,
             outfile: str | None = None) -> None:
    """Plot a simple bar chart.

    Useful for visualising:
    - jitter per method,
    - Cov_eps per method,
    - mean rank, mean error, etc.

    Parameters
    ----------
    categories : list of str
        Category labels (e.g. method names).
    values : list of float
        Values associated with each category.
    title : str
        Title of the plot.
    ylabel : str
        Label for the y-axis.
    rotation : int, optional
        Rotation angle for x-tick labels.
    outfile : str, optional
        If provided, save the figure instead of displaying it.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    idx = np.arange(len(categories))
    ax.bar(idx, values)
    ax.set_xticks(idx)
    ax.set_xticklabels(categories, rotation=rotation)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
        plt.close(fig)
    else:
        plt.show()


def plot_box(data: list[np.ndarray],
             labels: list[str],
             title: str,
             ylabel: str,
             outfile: str | None = None) -> None:
    """Plot a boxplot for comparing distributions across methods.

    Typical use: distribution of detection delays across multiple runs.

    Parameters
    ----------
    data : list of 1D ndarrays
        Each array contains samples for one method.
    labels : list of str
        Labels for each method.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    outfile : str, optional
        If provided, save the figure instead of displaying it.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
        plt.close(fig)
    else:
        plt.show()
