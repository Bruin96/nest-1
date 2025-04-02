"""Provides utilities for psychometric analysis."""

__version__ = "0.1"
__author__ = "Cara Tursun"
__copyright__ = """Copyright (c) 2022 Cara Tursun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
__license__ = "MIT"

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from .mle import logistic, weibull, logistic_mle_gda, weibull_mle_gda


def plot_staircase(
    levels: np.array,
    responses: np.array,
    converged_level: Optional[float] = None,
    reversalPoints: Optional[np.array] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Plot the stimulus intensity and responses for a
    staircase procedure.

    Parameters
    ----------
    levels : np.array
        An array with the stimulus intensities tested.

    responses : np.array
        An array of responses from the participant, 1 if
        detected and 0 if not detected.

    converged_level : float, optional
        The convergence value of the procedure.

    reversalPoints : np.array, optional
        An array containing the array indices of the
        reversal points in the parameter "levels".

    title : str, optional
        The title of the plot

    Returns
    -------
        Figure handle.

    """
    levels_unique = np.unique(levels)
    detection_rate = np.zeros(levels_unique.shape)
    num_measurements = np.zeros(levels_unique.shape)
    responses_det = np.zeros(levels_unique.shape)
    responses_nodet = np.zeros(levels_unique.shape)
    if converged_level is None:
        converged_level = levels[-1]

    # plot trials
    plt.figure(figsize=(15, 6))
    trial_idx = np.array(list(range(1, np.size(levels) + 1)))
    for level in levels_unique:
        idx = np.argwhere(levels == level)
        num_measurements = np.size(idx)
        responses_det = np.append(responses_det, np.sum(responses[idx]))
        responses_nodet = np.append(
            responses_nodet, num_measurements - responses_det
        )
        detection_rate = np.append(
            detection_rate, responses_det / num_measurements
        )
    # plot the line
    (line,) = plt.plot(
        trial_idx,
        levels,
        color="k",
        marker="None",
        linestyle="-",
        linewidth=1.5,
    )
    line.set_label("Answers")
    # plot detected levels
    (detected,) = plt.plot(
        trial_idx[np.argwhere(responses)],
        levels[np.argwhere(responses)],
        color="k",
        marker="s",
        markerfacecolor="k",
        linestyle="None",
        markeredgewidth=1.5,
        markersize=9,
    )
    detected.set_label("Detected")
    # plot undetected levels
    (not_detected,) = plt.plot(
        trial_idx[np.argwhere(responses == 0)],
        levels[np.argwhere(responses == 0)],
        color="k",
        marker="s",
        markerfacecolor="w",
        linestyle="None",
        markeredgewidth=1.5,
        markersize=9,
    )
    not_detected.set_label("Not detected")
    if reversalPoints is not None:
        (reversals,) = plt.plot(
            reversalPoints + 1,
            levels[reversalPoints],
            color="k",
            marker="o",
            markerfacecolor="None",
            linestyle="",
            linewidth=0,
            markeredgewidth=1.5,
            markersize=18,
        )
        reversals.set_label("Reversal $R_i$")
    # plot the stimulus level at convergence
    if converged_level is not None:
        plt.axhline(
            y=converged_level,
            color="k",
            linestyle=":",
            label="Converged level",
        )
    plt.grid(which="major", color="silver", linestyle="-", linewidth=0.5)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    ax = plt.gca()

    # if the number of trials are more than 15, show major tick
    # labels and grid lines only at multiples of 5.
    if np.max(trial_idx) >= 15:
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        plt.grid(which="minor", color="silver", linestyle="-", linewidth=0.25)
    else:
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("Trial Number", fontsize=15)
    ax.set_ylabel("Stimuli Intensity", fontsize=15)
    ax.legend(labelspacing=0.3, prop={"size": 15}, loc=[1.05, 0.25])
    if title is not None:
        plt.title(title, fontsize=15)
    plt.tight_layout()
    return plt.gcf()


def fit_psychometric(
    levels: np.array,
    responses: np.array,
    converged_level: Optional[float] = None,
    title: Optional[str] = None,
    beta_0: Optional[float] = None,
    guess_rate: Optional[float] = None,
    plot_results: Optional[bool] = False,
    psychometric_fn: Optional[str] = "logistic"
    ):
    """Fit a psychometric function to the results of a staircase
    procedure using MLE and plot it.

    Parameters
    ----------
    levels : np.array
        An array with the stimulus intensities tested.

    responses : np.array
        An array of responses from the participant, 1 if
        detected and 0 if not detected.

    converged_level : float, optional
        The convergence value of the procedure. It will
        be shown on the plots for comparison if provided.

    title : str, optional
        The title of the plot.

    Returns
    -------
        Figure handle.

    """
    if psychometric_fn == "logistic":
        mle_fn = logistic_mle_gda
        prob_fn = logistic
    elif psychometric_fn == "weibull":
        mle_fn = weibull_mle_gda
        prob_fn = weibull
    else:
        raise ValueError(f"Psychometric function {psychometric_fn} is not supported.")
    levels_unique = np.sort(np.unique(levels))
    detection_rate = np.zeros(levels_unique.shape)
    num_measurements = np.zeros(levels_unique.shape)
    responses_det = np.zeros(levels_unique.shape)
    responses_nodet = np.zeros(levels_unique.shape)
    for (i, level) in enumerate(levels_unique):
        num_measurements[i] = np.sum(levels == level)
        responses_det[i] = np.sum(responses[levels == level])
        responses_nodet[i] = num_measurements[i] - responses_det[i]
        detection_rate[i] = responses_det[i] / num_measurements[i]
    # Psychometric function fitting
    # result = logistic_mle(
    #     levels_unique,
    #     responses_det,
    #     responses_nodet,
    #     beta_0=beta_0,
    #     guess_rate=guess_rate,
    # )
    result = mle_fn(
        levels_unique,  # this should be in log domain for logistic, and linear domain for weibull
        responses_det,
        responses_nodet,
        beta_0=beta_0,
        guess_rate=guess_rate,
    )

    ## generate x-axis values for plotting
    if psychometric_fn == "logistic":
        x_levels = np.linspace(np.min(levels), np.max(levels), num=500)
    elif psychometric_fn == "weibull":
        x_levels = np.logspace(np.log10(np.min(levels)), np.log10(np.max(levels)), num=500)
    else:
        raise ValueError(f"Psychometric function {psychometric_fn} is not supported.")
    y_probabilities = prob_fn(
        x_levels,
        result["beta_0"],
        result["beta_1"],
        result["guessRate"],
        result["lapseRate"],
    )

    # to compute Rsq and Rsq_adj later
    predictions = prob_fn(
        levels_unique,
        result["beta_0"],
        result["beta_1"],
        result["guessRate"],
        result["lapseRate"],
    )
    result["predictions"] = predictions
    result["ground_truth"] = detection_rate

    if plot_results:
        plt.figure(figsize=(15, 6))
        levels_plot = None
        if psychometric_fn == "logistic":
            plt.plot(x_levels, y_probabilities, color="k", label=f"{psychometric_fn} MLE")
            levels_plot = levels_unique
            if converged_level is not None:
                converged_level_plot = converged_level
            beta0_plot = result["beta_0"]
        elif psychometric_fn == "weibull":
            plt.plot(np.log(x_levels), y_probabilities, color="k", label=f"{psychometric_fn} MLE")
            levels_plot = np.log(levels_unique)
            if converged_level is not None:
                converged_level_plot = np.log(converged_level)
            beta0_plot = np.log(result["beta_0"])
        else:
            raise ValueError(f"Psychometric function {psychometric_fn} is not supported.")

        plt.plot(
            levels_plot,
            detection_rate,
            color="k",
            marker="o",
            markerfacecolor="None",
            linestyle="",
            linewidth=0,
            markeredgewidth=1.5,
            markersize=18,
            label="Detection Rate",
        )
        # plot the stimulus level at convergence
        if converged_level is not None:
            plt.axvline(
                converged_level,
                color="k",
                linestyle=":",
                label="Converged level",
            )
        # plot the threshold estimated by the psychometric function fit
        plt.axvline(
            beta0_plot,
            color="k",
            linestyle="--",
            label="Estimated threshold",
        )
        plt.grid(which="major", color="silver", linestyle="-", linewidth=0.5)
        ax = plt.gca()
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("Stimuli Intensity", fontsize=15)
        ax.set_ylabel("Detection Rate", fontsize=15)
        ax.legend(labelspacing=0.3, prop={"size": 15}, loc=[1.05, 0.25])
        plt.title(title)
        plt.tight_layout()
        return (plt.gcf(), result)
    else:
        return (None, result)


def main():
    # this main function is used for testing purposes
    levels = np.array([64, 24, 32, 32] * 10)
    responses = np.array([1, 0, 1, 1] * 10)
    # reversalPoints = np.array([1])
    psychometric_fn = "weibull"
    (fig, result) = fit_psychometric(
        levels,
        responses,
        converged_level=32,
        title="title",
        plot_results=True,
        psychometric_fn=psychometric_fn)
    print(result)
    plt.show()


if __name__ == "__main__":
    main()
