from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator


class BaseEM(BaseEstimator, metaclass=ABCMeta):
    """Basic class for our Machine Learning Expectation-Maximization framework."""

    @abstractmethod
    def __init__(self, clustering, maximization,
                 stability_threshold, noise_tolerance_threshold,
                 n_consensus, n_iterations):

        # define hyperparameters such as stability_threshold or noise_tolerance_threshold
        assert (0 < stability_threshold <= 1), "The stability_threshold value is invalid. It must be between 0 and 1."
        assert (noise_tolerance_threshold > 1), "The noise_tolerance_threshold value is invalid. It must be at least 1."
        self.stability_threshold = stability_threshold
        self.noise_tolerance_threshold = noise_tolerance_threshold

        # define number of iterations or consensus to perform
        assert (type(n_iterations) is int and n_iterations > 1), "The number of iterations must be an integer, at least 2"
        assert (type(n_consensus) is int and n_consensus > 0), "The number of consensus must be an integer, at least 1"
        self.n_consensus = n_consensus
        self.n_iterations = n_iterations

        assert (clustering in ['k_means', 'full_gaussian_mixture', 'spherical_gaussian_mixture']), \
            "Clustering must be one of 'k_means', 'full_gaussian_mixture', 'spherical_gaussian_mixture'"
        self.clustering_method_name = clustering

        assert (maximization in ['linear', 'support_vector']), \
            "Maximization must be one of 'linear', 'support_vector'"
        self.maximization = maximization


