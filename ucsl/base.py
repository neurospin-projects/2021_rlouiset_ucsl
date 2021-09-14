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

        """# define what type of initialization, clustering, classification and consensus one wants to use
        assert (initialization in ['k_means', 'gaussian_mixture', 'DPP', 'precomputed']), \
            "Initialization must be one of 'k_means', 'gaussian_mixture', 'DPP', 'precomputed'"
        if initialization == 'precomputed' :
            assert (custom_initialization_matrixes is not None), \
                "if initialization is custom you have to pass a initialization_matrixes different from None"
            self.custom_initialization_matrixes = custom_initialization_matrixes
        self.initialization = initialization"""

        """assert (clustering in ['k_means', 'gaussian_mixture', 'HYDRA', 'custom']), \
            "Clustering must be one of 'k_means', 'gaussian_mixture', 'HYDRA', 'custom'"
        if clustering == 'custom' :
            assert (custom_clustering_method is not None), \
                "if clustering is custom you have to pass a custom_clustering_method different from None"
            self.custom_clustering_method = custom_clustering_method"""
        self.clustering = clustering

        """assert (maximization in ['max_margin', 'logistic', 'svr', 'custom']), \
            "maximization must be one of 'max_margin', 'logistic', 'svr', 'custom'"
        if maximization == 'custom' :
            assert (custom_maximization_method is not None), \
                "if maximization is custom you have to pass a custom_maximization_method different from None"
            self.custom_classification_method = custom_maximization_method"""
        self.maximization = maximization


