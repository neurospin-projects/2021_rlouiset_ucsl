import scipy
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
import numpy as np


def one_hot_encode(y, n_classes=None):
    ''' utils function in order to turn a label vector into a one hot encoded matrix '''
    if n_classes is None:
        n_classes = np.max(y) + 1
    y_one_hot = np.copy(y)
    return np.eye(n_classes)[y_one_hot]


def sigmoid(x, lambda_=5):
    return 1 / (1 + np.exp(-lambda_ * x))


def py_softmax(x, axis=None):
    """stable softmax"""
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def consensus_clustering(clustering_results, n_clusters, index_positives):
    S = np.ones((clustering_results.shape[0], n_clusters)) / n_clusters
    co_occurrence_matrix = np.zeros((clustering_results.shape[0], clustering_results.shape[0]))

    for i in range(clustering_results.shape[0] - 1):
        for j in range(i + 1, clustering_results.shape[0]):
            co_occurrence_matrix[i, j] = sum(clustering_results[i, :] == clustering_results[j, :])

    co_occurrence_matrix = np.add(co_occurrence_matrix, co_occurrence_matrix.transpose())
    # here is to compute the Laplacian matrix
    Laplacian = np.subtract(np.diag(np.sum(co_occurrence_matrix, axis=1)), co_occurrence_matrix)

    Laplacian_norm = np.subtract(np.eye(clustering_results.shape[0]), np.matmul(
        np.matmul(np.diag(1 / np.sqrt(np.sum(co_occurrence_matrix, axis=1))), co_occurrence_matrix),
        np.diag(1 / np.sqrt(np.sum(co_occurrence_matrix, axis=1)))))
    # replace the nan with 0
    Laplacian_norm = np.nan_to_num(Laplacian_norm)

    # check if the Laplacian norm is symmetric or not, because matlab eig function will automatically check this, but not in numpy or scipy
    e_value, e_vector = scipy.linalg.eigh(Laplacian_norm)

    # check if the eigen vector is complex
    if np.any(np.iscomplex(e_vector)):
        e_value, e_vector = scipy.linalg.eigh(Laplacian)

    # train Spectral Clustering algorithm and make predictions
    spectral_features = e_vector.real[:, :n_clusters]

    # apply clustering method
    k_means = KMeans(n_clusters=n_clusters).fit(spectral_features[index_positives])
    S[index_positives] = one_hot_encode(k_means.labels_.astype(np.int), n_classes=n_clusters)

    return S


def compute_similarity_matrix(consensus_assignment, clustering_assignments_to_pred=None):
    # compute inter-samples positive/negative co-occurence matrix
    similarity_matrix = np.zeros((len(consensus_assignment), len(clustering_assignments_to_pred)))
    for i, p_assignment in enumerate(consensus_assignment):
        for j, new_point_assignment in enumerate(clustering_assignments_to_pred):
            similarity_matrix[i, j] = np.sum(p_assignment == new_point_assignment)
    similarity_matrix += 1e-3
    similarity_matrix /= np.max(similarity_matrix)
    return similarity_matrix


def compute_spectral_clustering_consensus(clustering_results, n_clusters):
    # compute positive samples co-occurence matrix
    n_positives = len(clustering_results)
    similarity_matrix = np.zeros((n_positives, n_positives))
    for i in range(n_positives - 1):
        for j in range(i + 1, n_positives):
            similarity_matrix[i, j] = sum(clustering_results[i, :] == clustering_results[j, :])
    similarity_matrix = np.add(similarity_matrix, similarity_matrix.transpose())
    similarity_matrix += 1e-3
    similarity_matrix /= np.max(similarity_matrix)

    # initialize spectral clustering method
    spectral_clustering_method = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    spectral_clustering_method.fit(similarity_matrix)

    return spectral_clustering_method.labels_


def launch_svc(X, y, sample_weight=None, kernel='linear', C=1):
    """Fit the classification SVMs according to the given training data.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training vectors.
    y : array-like, shape (n_samples,)
        Target values.
    sample_weight : array-like, shape (n_samples,)
        Training sample weights.
    kernel : string,
        kernel used for SVM.
    C : float,
        SVM hyperparameter C
    Returns
    -------
    SVM_coefficient : array-like, shape (1, n_features)
        The coefficient of the resulting SVM.
    SVM_intercept : array-like, shape (1,)
        The intercept of the resulting SVM.
    """

    # fit the different SVM/hyperplanes
    SVM_classifier = SVC(kernel=kernel, C=C)
    SVM_classifier.fit(X, y, sample_weight=sample_weight)

    # get SVM intercept value
    SVM_intercept = SVM_classifier.intercept_

    # get SVM hyperplane coefficient
    SVM_coefficient = SVM_classifier.coef_

    return SVM_coefficient, SVM_intercept

def launch_svr(X, y, sample_weight=None, kernel='linear', C=1):
    """Fit the classification SVMs according to the given training data.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training vectors.
    y : array-like, shape (n_samples,)
        Target values.
    sample_weight : array-like, shape (n_samples,)
        Training sample weights.
    kernel : string,
        kernel used for SVM.
    C : float,
        SVM hyperparameter C
    Returns
    -------
    SVM_coefficient : array-like, shape (1, n_features)
        The coefficient of the resulting SVM.
    SVM_intercept : array-like, shape (1,)
        The intercept of the resulting SVM.
    """

    # fit the different SVM/hyperplanes
    SVM_regressor = SVR(kernel=kernel, C=C)
    SVM_regressor.fit(X, y, sample_weight=sample_weight)

    # get SVM intercept value
    SVM_intercept = SVM_regressor.intercept_

    # get SVM hyperplane coefficient
    SVM_coefficient = SVM_regressor.coef_

    return SVM_coefficient, SVM_intercept


def launch_logistic(X, y, sample_weight=None):
    """Fit the logistic regressions according to the given training data.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training vectors.
    y : array-like, shape (n_samples,)
        Target values.
    sample_weight : array-like, shape (n_samples,)
        Training sample weights.
    Returns
    -------
    logistic_coefficient : array-like, shape (1, n_features)
        The coefficient of the resulting logistic regression.
    """

    # fit the different logistic classifier
    logistic = LogisticRegression(max_iter=200)
    logistic.fit(X, y, sample_weight=sample_weight)

    # get logistic coefficient and intercept
    logistic_coefficient = logistic.coef_
    logistic_intercept = logistic.intercept_

    return logistic_coefficient, logistic_intercept

def launch_linear(X, y, sample_weight=None):
    """Fit the linear regressions according to the given training data.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training vectors.
    y : array-like, shape (n_samples,)
        Target values.
    sample_weight : array-like, shape (n_samples,)
        Training sample weights.
    Returns
    -------
    logistic_coefficient : array-like, shape (1, n_features)
        The coefficient of the resulting logistic regression.
    """

    # fit the different linear regressor
    linear = LinearRegression()
    linear.fit(X, y, sample_weight=sample_weight)

    # get logistic coefficient and intercept
    logistic_coefficient = linear.coef_
    logistic_intercept = linear.intercept_

    return logistic_coefficient, logistic_intercept
