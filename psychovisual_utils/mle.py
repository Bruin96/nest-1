"""Provides utilities for maximum likelihood estimate of
psychometric functions."""

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

import numpy as np
from scipy.optimize import minimize


def logistic(
    x: np.array,
    beta_0: float,
    beta_1: float,
    guessRate: float,
    lapseRate: float,
) -> np.array:
    return guessRate + (1 / (1 + np.exp(-(x - beta_0) * beta_1))) * (1 - guessRate) * (1 - lapseRate)


def weibull(
    x: np.array,
    beta_0: float,
    beta_1: float,
    guessRate: float,
    lapseRate: float,
) -> np.array:
    # return ((1 - np.exp(-np.power(x / beta_0, beta_1)) * (1 - guessRate)) - guessRate) * lapseRate + guessRate
    return (guessRate - 1) * (np.exp(-np.power(x / beta_0, beta_1)) - 1) * (1-lapseRate) + guessRate


def logistic_likelihood(
    levels: np.array,
    responses_det: np.array,
    responses_nodet: np.array,
    beta_0: float,
    beta_1: float,
    guessRate: float,
    lapseRate: float,
) -> float:
    likelihood = 1
    for n in range(levels.size):
        p = logistic(levels[n], beta_0, beta_1, guessRate, lapseRate)
        likelihood *= (p ** responses_det[n]) * (
            (1 - p) ** (responses_nodet[n])
        )
    return likelihood


def weibull_likelihood(
    levels: np.array,
    responses_det: np.array,
    responses_nodet: np.array,
    beta_0: float,
    beta_1: float,
    guessRate: float,
    lapseRate: float,
) -> float:
    likelihood = 1
    for n in range(levels.size):
        p = weibull(levels[n], beta_0, beta_1, guessRate, lapseRate)
        likelihood *= (p ** responses_det[n]) * (
            (1 - p) ** (responses_nodet[n])
        )
    return likelihood


def logistic_loglikelihood(
    levels: np.array,
    responses_det: np.array,
    responses_nodet: np.array,
    beta_0: float,
    beta_1: float,
    guessRate: float,
    lapseRate: float,
) -> float:
    loglikelihood = np.array(0, dtype=np.float32)
    epsilon = 1e-3
    for n in range(levels.size):
        p = logistic(levels[n], beta_0, beta_1, guessRate, lapseRate)
        ll = 0
        if responses_det[n] > 0:
            ll += np.log(np.maximum(p, epsilon)) * responses_det[n]
        if responses_nodet[n] > 0:
            ll += np.log(np.maximum(1 - p, epsilon)) * responses_nodet[n]
        loglikelihood += ll
    return loglikelihood


def weibull_loglikelihood(
    levels: np.array,
    responses_det: np.array,
    responses_nodet: np.array,
    beta_0: float,
    beta_1: float,
    guessRate: float,
    lapseRate: float,
) -> float:
    epsilon = 1e-3
    loglikelihood = np.array(0, dtype=np.float32)
    for n in range(levels.size):
        p = weibull(levels[n], beta_0, beta_1, guessRate, lapseRate)
        ll = 0
        if responses_det[n] > 0:
            ll += np.log(np.maximum(p, epsilon)) * responses_det[n]
        if responses_nodet[n] > 0:
            ll += np.log(np.maximum(1 - p, epsilon)) * responses_nodet[n]
        loglikelihood += ll
    return loglikelihood


def logistic_loglikelihood_jacobian(
    levels: np.array,
    responses_det: np.array,
    responses_nodet: np.array,
    beta_0: float,
    beta_1: float,
    guessRate: float,
    lapseRate: float,
) -> np.array:
    def ll_delbeta_0(x): return - beta_1 * (guessRate-1) * (lapseRate-1) * \
        np.exp(beta_1 * (x-beta_0)) / \
        np.square(1 + np.exp(beta_1 * (x - beta_0)))

    def ll_delbeta_1(x): return (guessRate-1) * (lapseRate-1) * (x-beta_0) * \
        np.exp(beta_1 * (x-beta_0)) / \
        np.square(1 + np.exp(beta_1 * (x - beta_0)))
    def ll_dellapseRate(x): return - (1-guessRate) / \
        (1 + np.exp(-beta_1 * (x-beta_0)))
    # def ll_delguessRate(x): return 1 - (1-lapseRate) / \
    #     (1 + np.exp(-beta_1 * (x-beta_0)))
    probs = logistic(levels, beta_0, beta_1, guessRate, lapseRate)
    # partial derivative of log-likelihood w.r.t. prob. of det from logistic
    # function for each level of stimulus
    # ll_delprobs = (probs - 1) / ((probs-1) * probs) * responses_det + \
    #     (probs) / ((probs-1) * probs) * responses_nodet
    epsilon = 1e-3
    ll_delprobs = responses_det / np.maximum(probs, epsilon) - \
        responses_nodet / np.maximum(1 - probs, epsilon)
    grads = np.array(
        [
            np.sum(ll_delbeta_0(levels) * ll_delprobs),
            np.sum(ll_delbeta_1(levels) * ll_delprobs),
            np.sum(ll_dellapseRate(levels) * ll_delprobs)]
    )
    return grads


def weibull_loglikelihood_jacobian(
    levels: np.array,
    responses_det: np.array,
    responses_nodet: np.array,
    beta_0: float,
    beta_1: float,
    guessRate: float,
    lapseRate: float,
) -> np.array:
    def probs_delbeta_0(x): return beta_1 * (guessRate - 1) * \
        (1 - lapseRate) * np.power(x / beta_0, beta_1) * \
        np.exp(-np.power(x / beta_0, beta_1)) / beta_0

    def probs_delbeta_1(x): return - (guessRate - 1) * \
        (1 - lapseRate) * np.power(x / beta_0, beta_1) * \
        np.log(x / beta_0) * np.exp(-np.power(x / beta_0, beta_1))

    def probs_dellapseRate(x): return -(guessRate - 1) * \
        (np.exp(-np.power(x / beta_0, beta_1)) - 1)

    # def ll_delguessRate(x): return (1 - lapseRate) * \
    #     (np.exp(-np.power(x / beta_0, beta_1)) - 1) + 1

    probs = weibull(levels, beta_0, beta_1, guessRate, lapseRate)
    # partial derivative of log-likelihood w.r.t. prob. of det from logistic
    # function for each level of stimulus
    # ll_delprobs = (probs - 1) / ((probs-1) * probs) * responses_det + \
    #     (probs) / ((probs-1) * probs) * responses_nodet
    epsilon = 1e-3
    ll_delprobs = responses_det / np.maximum(probs, epsilon) - \
        responses_nodet / np.maximum(1 - probs, epsilon)
    grads = np.array(
        [
            np.sum(probs_delbeta_0(levels) * ll_delprobs),
            np.sum(probs_delbeta_1(levels) * ll_delprobs),
            np.sum(probs_dellapseRate(levels) * ll_delprobs)]
    )
    return grads

def logistic_mle_gda(
    levels: np.array,
    responses_det: np.array,
    responses_nodet: np.array,
    beta_0: Optional[float] = None,
    guess_rate: Optional[float] = None,
    lapse_rate: Optional[float] = None,
) -> dict:
    """ Logistic MLE using Gradient Ascent """
    constraints = []
    if beta_0 is None:
        beta_0 = 0.0
        constraints.append({})
    else:
        constraints.append({"eq": beta_0})
    constraints.append({})  # no constraints on beta_1
    if guess_rate is None:
        guess_rate = 0.5
    if lapse_rate is None:
        lapse_rate = 0.0
        constraints.append({})
    else:
        constraints.append({"eq": lapse_rate})
    beta_1 = 1
    x0 = [beta_0, beta_1, lapse_rate]
    bounds = [(np.min(levels), np.max(levels)), (0, float("inf")), (0, 0.05)]
    reg = 0.00001  # regularization coefficient

    def fun(x):
        return -logistic_loglikelihood(
            levels,
            responses_det,
            responses_nodet,
            x[0],
            x[1],
            guess_rate,
            x[2]) + 0.5 * reg * (np.square(x[0]) + np.square(x[1]) + np.square(x[2]))

    def fun_noreg(x):
        return -logistic_loglikelihood(
            levels,
            responses_det,
            responses_nodet,
            x[0],
            x[1],
            guess_rate,
            x[2])

    def jac_fun(x):
        jac = -logistic_loglikelihood_jacobian(
            levels,
            responses_det,
            responses_nodet,
            x[0],
            x[1],
            guess_rate,
            x[2])
        jac = [(j + reg * e) for (j, e) in zip(jac, x)]
        return jac

    # jac = jac_fun(x0)
    # epsilon = 1e-3
    # orig = fun(x0)
    # jac_num = [
    #     (fun([beta_0 + epsilon, beta_1, lapse_rate]) - orig) / epsilon,
    #     (fun([beta_0, beta_1 + epsilon, lapse_rate]) - orig) / epsilon,
    #     (fun([beta_0, beta_1, lapse_rate + epsilon]) - orig) / epsilon
    # ]
    # optimizers = ["CG", "BFGS", "Newton-CG", "L-BFGS-B",
    #               "TNC", "SLSQP", "trust-exact", "trust-constr"]
    # for optimizer in optimizers:
    #     res = minimize(fun, x0, jac=jac_fun, method=optimizer)
    #     print(f"{optimizer} {res['message']}")
    res = minimize(fun, x0, jac=jac_fun, method="SLSQP", bounds=bounds)
    x = res["x"]
    result = {
        "beta_0": x[0],
        "beta_1": x[1],
        "guessRate": guess_rate,
        "lapseRate": x[2],
        "likelihood": -fun_noreg(x)  # this is actually log-likelihood
    }
    return result


def weibull_mle_gda(
    levels: np.array,
    responses_det: np.array,
    responses_nodet: np.array,
    beta_0: Optional[float] = None,
    guess_rate: Optional[float] = None,
    lapse_rate: Optional[float] = None,
) -> dict:
    """ Weibull MLE using Gradient Ascent """
    constraints = []
    if beta_0 is None:
        beta_0 = 0.1
        constraints.append({})
    else:
        constraints.append({"eq": beta_0})
    constraints.append({})  # no constraints on beta_1
    if guess_rate is None:
        guess_rate = 0.5
    if lapse_rate is None:
        lapse_rate = 0.0
        constraints.append({})
    else:
        constraints.append({"eq": lapse_rate})
    beta_1 = 1.0
    x0 = [beta_0, beta_1, lapse_rate]
    bounds = [(np.min(levels), np.max(levels)), (0, float("inf")), (0, 0.05)]
    reg = 0.00001  # regularization coefficient

    def fun(x):
        return -weibull_loglikelihood(
            levels,
            responses_det,
            responses_nodet,
            x[0],
            x[1],
            guess_rate,
            x[2]) + 0.5 * reg * (np.square(x[0]) + np.square(x[1]) + np.square(x[2]))

    def fun_noreg(x):
        return -weibull_loglikelihood(
            levels,
            responses_det,
            responses_nodet,
            x[0],
            x[1],
            guess_rate,
            x[2])

    def jac_fun(x):
        jac = -weibull_loglikelihood_jacobian(
            levels,
            responses_det,
            responses_nodet,
            x[0],
            x[1],
            guess_rate,
            x[2])
        jac = [(j + reg * e) for (j, e) in zip(jac, x)]
        return jac

    # jac = jac_fun(x0)
    # def jac_num(x0, epsilon):
    #     orig = fun(x0)
    #     beta_0, beta_1, lapse_rate = x0
    #     numerical = [
    #         (fun([beta_0 + epsilon, beta_1, lapse_rate]) - orig) / epsilon,
    #         (fun([beta_0, beta_1 + epsilon, lapse_rate]) - orig) / epsilon,
    #         (fun([beta_0, beta_1, lapse_rate + epsilon]) - orig) / epsilon
    #     ]
    #     return numerical
    # optimizers = ["CG", "BFGS", "Newton-CG", "L-BFGS-B",
    #               "TNC", "SLSQP"] #, "trust-exact", "trust-constr"]
    # for optimizer in optimizers:
    #     # res = minimize(fun, x0, jac=lambda x : jac_num(x, 1e-4), method=optimizer)
    #     res = minimize(fun, x0, jac=jac_fun, method=optimizer)
    #     print(f"{optimizer} {res['message']}")
    #     print(res['x'])

    res = minimize(fun, x0, jac=jac_fun, method="SLSQP", bounds=bounds)
    x = res["x"]
    result = {
        "beta_0": x[0],
        "beta_1": x[1],
        "guessRate": guess_rate,
        "lapseRate": x[2],
        "likelihood": -fun_noreg(x)  # this is actually log-likelihood
    }
    return result

# def logistic_mle(
#     levels: np.array,
#     responses_det: np.array,
#     responses_nodet: np.array,
#     beta_0: Optional[float] = None,
#     guess_rate: Optional[float] = None,
#     lapse_rate: Optional[float] = None,
# ) -> dict:
#     """ Logistic psychometric function MLE using grid search. """
#     if beta_0 is not None:
#         beta_0_min_max_step = (beta_0, beta_0, 1)
#     else:
#         beta_0_min_max_step = (np.min(levels), np.max(levels), 50)
#     beta_1_min_max_step = (np.log10(0.01), np.log10(100), 50)
#     if guess_rate is not None:
#         guessRate_min_max_step = (guess_rate, guess_rate, 1)
#     else:
#         guessRate_min_max_step = (0.0, 0.50, 5)
#     if lapse_rate is not None:
#         lapseRate_min_max_step = (lapse_rate, lapse_rate, 1)
#     else:
#         lapseRate_min_max_step = (0, 0.20, 5)
#     max_likelihood = np.array(0, dtype=np.float32)
#     beta_0_opt, beta_1_opt, guessRate_opt, lapseRate_opt = 0, 0, 0, 0
#     for beta_0 in np.linspace(
#         beta_0_min_max_step[0],
#         beta_0_min_max_step[1],
#         num=beta_0_min_max_step[2],
#     ):
#         for beta_1 in np.logspace(
#             beta_1_min_max_step[0],
#             beta_1_min_max_step[1],
#             num=beta_1_min_max_step[2],
#         ):
#             for guessRate in np.linspace(
#                 guessRate_min_max_step[0],
#                 guessRate_min_max_step[1],
#                 num=guessRate_min_max_step[2],
#             ):
#                 for lapseRate in np.linspace(
#                     lapseRate_min_max_step[0],
#                     lapseRate_min_max_step[1],
#                     num=lapseRate_min_max_step[2],
#                 ):
#                     likelihood = logistic_loglikelihood(
#                         levels,
#                         responses_det,
#                         responses_nodet,
#                         beta_0,
#                         beta_1,
#                         guessRate,
#                         lapseRate,
#                     )
#                     if max_likelihood < likelihood:
#                         max_likelihood = likelihood
#                         (
#                             beta_0_opt,
#                             beta_1_opt,
#                             guessRate_opt,
#                             lapseRate_opt,
#                         ) = (beta_0, beta_1, guessRate, lapseRate)
#     result = {
#         "likelihood": max_likelihood,
#         "beta_0": beta_0_opt,
#         "beta_1": beta_1_opt,
#         "guessRate": guessRate_opt,
#         "lapseRate": lapseRate_opt,
#     }
#     return result


def main():
    import matplotlib.pyplot as plt
    x = np.linspace(0, 10, num=100)
    beta_0 = 5
    beta_1 = 3.0
    guessRate = 0.5
    lapseRate = 0.25
    p = weibull(x, beta_0, beta_1, guessRate, lapseRate)
    plt.plot(x,p)
    plt.show()


if __name__ == "__main__":
    main()
