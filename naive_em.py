"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n = X.shape[0]
    d = X.shape[1]
    mu = mixture.mu
    var = mixture.var
    p = mixture.p
    gaussian_denominator = 1/(2 * var * np.pi) ** (d / 2)
    gaussian_exponent = np.exp(- np.linalg.norm(X[:, np.newaxis] - mu, axis=2)**2 / (2 * var))
    soft_counts = p * gaussian_denominator * gaussian_exponent
    total_counts = soft_counts.sum(axis=1).reshape(n, 1)
    weighted_soft_counts = np.divide(soft_counts, total_counts)
    loglike = np.sum(np.log(total_counts), axis=0)
    return weighted_soft_counts, float(loglike)


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n = X.shape[0]
    d = X.shape[1]
    k = post.shape[1]
    n_hat = np.sum(post, axis = 0)
    p_hat = n_hat / n
    mu_hat = post.T @ X / (n_hat.reshape(k, 1))
    norm = np.linalg.norm(X[:, np.newaxis] - mu_hat, axis=2)**2
    summ = np.sum(post * norm, axis=0)
    var_hat = summ / (n_hat * d)
    return GaussianMixture(mu_hat, var_hat, p_hat)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """

    old_likelihood = 0
    likelihood = 0
    while (old_likelihood ==0 or likelihood - old_likelihood >= 1e-6 * np.abs(likelihood)):
        old_likelihood = likelihood
        post, likelihood = estep(X, mixture)
        mixture = mstep(X, post)

    return  mixture, post, likelihood
