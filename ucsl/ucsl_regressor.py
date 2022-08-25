import copy
import logging

from sklearn.base import RegressorMixin
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.mixture import GaussianMixture

from ucsl.base import *
from ucsl.utils import *


class UCSL_R(BaseEM, RegressorMixin):
    """ucsl regressor.
    Implementation of Mike Tipping"s Relevance Vector Machine for
    classification using the scikit-learn API.

    Parameters
    ----------
    clustering : string, optional (default="spherical_gaussian_mixture")
        Clustering method for the Expectation step,
        If not specified, "spherical_gaussian_mixture" (spherical by default) will be used.
        It must be one of "spherical_gaussian_mixture", "full_gaussian_mixture", "k-means"

    maximization ; string or object, optional (default="lr")
        Classification method for the maximization step,
        If not specified, "linear" (Linear Regression) will be used.
        It must be one of "linear", "support_vector"
        It can also be a sklearn-like object with fit and predict methods; coef_ and intercept_ attributes.

    n_clusters : int, optional (default=2)
        numbers of subtypes we are assuming (equal to K in UCSL original paper)
        If not specified, the value of 2 will be used.
        Must be > 1.

    n_iterations : int, optional (default=10)
        numbers of Expectation-Maximization step performed per consensus run
        If not specified, the value of 10 will be used.
        Must be > 1.

    n_consensus : int, optional (default=10)
        numbers of Expectation-Maximization loops performed before ensembling of all the clusterings
        If not specified, the value of 10 will be used.
        Must be > 1.

    stability_threshold : float, optional (default=0.9)
        Adjusted rand index threshold between 2 successive iterations clustering
        If not specified, the value of 0.9 will be used.
        Must be between 0 and 1.

    noise_tolerance_threshold : float, optional (default=10)
        Threshold tolerance in graam-schmidt algorithm
        Given an orthogonalized vector, if its norm is inferior to 1 / noise_tolerance_threshold,
        we do not add it to the orthonormalized basis.
        Must be > 0.
    """

    def __init__(self, stability_threshold=0.9, noise_tolerance_threshold=10,
                 n_consensus=10, n_iterations=10, n_clusters=2,
                 clustering='spherical_gaussian_mixture', maximization='linear'):

        super().__init__(clustering=clustering, maximization=maximization,
                         stability_threshold=stability_threshold, noise_tolerance_threshold=noise_tolerance_threshold,
                         n_consensus=n_consensus, n_iterations=n_iterations)

        # define the number of clusters needed
        self.n_clusters = n_clusters

        # store directions from the Maximization method and store intercepts
        self.coefficients = {cluster_i: [] for cluster_i in range(self.n_clusters)}
        self.intercepts = {cluster_i: [] for cluster_i in range(self.n_clusters)}

        # store intermediate and consensus results in dictionaries
        self.cluster_labels_ = None
        self.clustering_assignments = None

        # define barycenters saving dictionaries
        self.barycenters = None

        # define orthonormal directions basis and clustering methods at each consensus step
        self.orthonormal_basis = {c: {} for c in range(n_consensus)}
        self.clustering_method = {c: {} for c in range(n_consensus)}

    def fit(self, X_train, y_train):
        """Fit the ucsl model according to the given training data.
        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
            Training vectors.
        y_train : array-like, shape (n_samples,)
            Target values.
        Returns
        -------
        self
        """

        # run the algorithm
        self.run(X_train, y_train)

        return self

    def predict(self, X):
        """Predict using the ucsl model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Query points to be evaluate.
        Returns
        -------
        y_pred_regression : array, shape (n_samples,)
            Predictions of the labels of the query points.
        y_pred_clusters : array, shape (n_samples,)
            Predictions of the clusters of the query points.
        """
        y_pred_proba_clusters = self.predict_clusters(X)

        regression_results_per_clusters = self.regress_new_points(X)

        y_pred_regression = sum( [y_pred_proba_clusters[:, cluster] * regression_results_per_clusters[:, cluster] for cluster in range(self.n_clusters)])

        y_pred_clusters = np.argmax(y_pred_proba_clusters, 1)

        return y_pred_regression, y_pred_clusters

    def predict_proba(self, X):
        """Predict using the ucsl model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Query points to be evaluate.
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predictions of the labels of the query points.
        """
        y_pred_proba_clusters = self.predict_clusters(X)

        regression_results_per_clusters = self.regress_new_points(X)

        y_pred_regression = sum([y_pred_proba_clusters[:, cluster] * regression_results_per_clusters[:, cluster] for cluster in range(self.n_clusters)])

        return y_pred_regression, y_pred_proba_clusters

    def regress_new_points(self, X):
        """Predict using the ucsl model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Query points to be evaluate.
        Returns
        -------
        SVM_distances : dict of array, length (n_labels) , shape of element (n_samples, n_clusters[label])
            Predictions of the point/hyperplane margin for each cluster of each label.
        """
        # first compute points distances to hyperplane
        regression_results = np.zeros((len(X), self.n_clusters))

        for cluster_i in range(self.n_clusters):
            SVM_coefficient = self.coefficients[cluster_i]
            SVM_intercept = self.intercepts[cluster_i]
            regression_results[:, cluster_i] = X @ SVM_coefficient + SVM_intercept

        return regression_results

    def predict_clusters(self, X):
        """Predict clustering for each label in a hierarchical manner.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.
        Returns
        -------
        cluster_predictions : dict of arrays, length (n_labels) , shape per key:(n_samples, n_clusters[key])
            Dict containing clustering predictions for each label, the dictionary keys are the labels
        """
        X_proj = X @ self.orthonormal_basis[-1].T

        if self.clustering_method_name == "k_means":
            Q_distances = np.zeros((len(X_proj), self.n_clusters))
            for cluster in range(self.n_clusters):
                Q_distances[:, cluster] = np.sum((X_proj - self.barycenters[cluster][None, :])**2, 1)
            Q_distances = 1 / (Q_distances + 1e-5)
            y_pred_proba_clusters = Q_distances / np.sum(Q_distances, 1)[:, None]
        elif self.clustering_method_name in ['full_gaussian_mixture', 'spherical_gaussian_mixture'] :
            y_pred_proba_clusters = self.clustering_method[-1].predict_proba(X_proj)
        else:
            return NotImplementedError

        return y_pred_proba_clusters

    def run(self, X, y):
        n_consensus = self.n_consensus
        # define the clustering assignment matrix (each column correspond to one consensus run)
        self.clustering_assignments = np.zeros((len(X), n_consensus))

        for consensus in range(n_consensus):
            # first we initialize the clustering matrix S, with the initialization strategy set in self.initialization
            S, cluster_index = self.initialize_clustering(X, y)

            cluster_index = self.run_EM(X, y, S, cluster_index, consensus)

            # update the cluster index for the consensus clustering
            self.clustering_assignments[:, consensus] = cluster_index

        if n_consensus > 1:
            self.clustering_ensembling(X, y)

    def initialize_clustering(self, X, y_polytope):
        """Perform a bagging of the previously obtained clusterings and compute new hyperplanes.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.
        y_polytope : array-like, shape (n_samples,)
            Target values.
        Returns
        -------
        S : array-like, shape (n_samples, n_samples)
            Cluster prediction matrix.
        """
        S = np.ones((len(y_polytope), self.n_clusters)) / self.n_clusters

        if self.clustering_method_name in ["k_means"]:
            KM = KMeans(n_clusters=self.n_clusters, init="random", n_init=1).fit(X)
            KM_barycenters = KM.cluster_centers_
            for cluster in range(self.n_clusters):
                S[:, cluster] = np.sum((X - KM_barycenters[cluster])**2, 1)
            S = 1 / (S + 1e-5)
            S = S / np.sum(S, 1)[:, None]

        elif self.clustering_method_name == "spherical_gaussian_mixture":
            GMM = GaussianMixture(n_components=self.n_clusters, init_params="random", n_init=1, covariance_type="spherical").fit(X)
            S = GMM.predict_proba(X)

        elif self.clustering_method_name == "full_gaussian_mixture":
            GMM = GaussianMixture(n_components=self.n_clusters, init_params="random", n_init=1, covariance_type="full").fit(X)
            S = GMM.predict_proba(X)
            
        else :
            return NotImplementedError

        cluster_index = np.argmax(S, axis=1)

        return S, cluster_index

    def maximization_step(self, X, y, S, iteration):
        if self.maximization == "linear":
            for cluster in range(self.n_clusters):
                cluster_assignment = np.ascontiguousarray(S[:, cluster])
                logistic_coefficient, logistic_intercept = launch_linear(X, y, cluster_assignment)
                self.coefficients[cluster] = logistic_coefficient
                self.intercepts[cluster] = logistic_intercept
        elif self.maximization == "support_vector":
            for cluster in range(self.n_clusters):
                cluster_assignment = np.ascontiguousarray(S[:, cluster])
                logistic_coefficient, logistic_intercept = launch_svr(X, y, cluster_assignment)
                self.coefficients[cluster] = logistic_coefficient
                self.intercepts[cluster] = logistic_intercept
        else:
            for cluster in range(self.n_clusters):
                cluster_assignment = np.ascontiguousarray(S[:, cluster])
                self.maximization.fit(X, y, sample_weight=cluster_assignment)
                self.coefficients[cluster] = self.maximization.coef_
                self.intercepts[cluster] = self.maximization.intercept_

    def expectation_step(self, X, S, consensus):
        """Update clustering method (update clustering distribution matrix S).
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.

        S : array-like, shape (n_samples, n_samples)
            Cluster prediction matrix.
        consensus : int
            which consensus is being run ?
        Returns
        -------
        S : array-like, shape (n_samples, n_samples)
            Cluster prediction matrix.
        cluster_index : array-like, shape (n_positives_samples, )
            clusters predictions argmax for positive samples.
        """
        # get directions basis
        directions_basis = []
        for cluster in range(self.n_clusters):
            directions_basis.append(self.coefficients[cluster])
        norm_directions = [np.linalg.norm(direction) for direction in directions_basis]
        directions_basis = np.array(directions_basis) / np.array(norm_directions)[:, None]

        # apply graam-schmidt algorithm
        orthonormalized_basis = self.graam_schmidt(directions_basis)
        self.orthonormal_basis[consensus] = orthonormalized_basis
        self.orthonormal_basis[-1] = np.array(orthonormalized_basis).copy()
        X_proj = X @ self.orthonormal_basis[consensus].T

        # get centroids or barycenters
        centroids = [np.mean(S[:, cluster][:, None] * X_proj, 0) for cluster in range(self.n_clusters)]

        if self.clustering_method_name == 'k_means':
            self.clustering_method[consensus] = KMeans(n_clusters=self.n_clusters, init=np.array(centroids), n_init=1).fit(X_proj)
            KM_barycenters = self.clustering_method[consensus].cluster_centers_
            Q = np.ones((len(X_proj), self.n_clusters)) / self.n_clusters
            for cluster in range(self.n_clusters):
                Q[:, cluster] = np.sum((X_proj - KM_barycenters[cluster][None, :])**2, 1)
            Q = 1 / (Q + 1e-5)
            Q = Q / np.sum(Q, 1)[:, None]
        elif self.clustering_method_name == 'spherical_gaussian_mixture':
            self.clustering_method[consensus] = GaussianMixture(n_components=self.n_clusters, covariance_type="spherical", means_init=np.array(centroids)).fit(X_proj)
            Q = self.clustering_method[consensus].predict_proba(X_proj)
            self.clustering_method[-1] = copy.deepcopy(self.clustering_method[consensus])

        elif self.clustering_method_name == 'full_gaussian_mixture':
            self.clustering_method[consensus] = GaussianMixture(n_components=self.n_clusters, covariance_type="full", means_init=np.array(centroids)).fit(X_proj)
            Q = self.clustering_method[consensus].predict_proba(X_proj)
            self.clustering_method[-1] = copy.deepcopy(self.clustering_method[consensus])
        else:
            return NotImplementedError

        # define matrix clustering
        S = Q.copy()
        cluster_index = np.argmax(Q, axis=1)

        return S, cluster_index

    def graam_schmidt(self, directions_basis):
        # compute the most important vectors because Graam-Schmidt is not invariant by permutation when the matrix is not square
        scores = []
        for i, direction_i in enumerate(directions_basis):
            scores_i = []
            for j, direction_j in enumerate(directions_basis):
                if i != j:
                    scores_i.append(np.linalg.norm(direction_i - (np.dot(direction_i, direction_j) * direction_j)))
            scores.append(np.mean(scores_i))
        directions = directions_basis[np.array(scores).argsort()[::-1], :]

        # orthonormalize coefficient/direction basis
        basis = []
        for v in directions:
            w = v - np.sum(np.dot(v, b) * b for b in basis)
            if len(basis) >= 2:
                if np.linalg.norm(w) * self.noise_tolerance_threshold > 1:
                    basis.append(w / np.linalg.norm(w))
            elif np.linalg.norm(w) > 1e-2:
                basis.append(w / np.linalg.norm(w))
        return np.array(basis)

    def run_EM(self, X, y, S, cluster_index, consensus):
        """Perform a bagging of the previously obtained clustering and compute new hyperplanes.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.
        y : array-like, shape (n_samples,)
            Original target values.
        S : array-like, shape (n_samples, n_samples)
            Cluster prediction matrix.
        cluster_index : array-like, shape (n_positives_samples, )
            clusters predictions argmax for positive samples.
        consensus : int
            index of consensus
        Returns
        -------
        S : array-like, shape (n_samples, n_samples)
            Cluster prediction matrix.
        """
        best_cluster_consistency = 1
        if consensus == -1:
            consensus = self.n_consensus + 1
            best_cluster_consistency = 0

        for iteration in range(self.n_iterations):
            # check for degenerate clustering for positive labels (warning) and negatives (might be normal)
            for cluster in range(self.n_clusters):
                if np.count_nonzero(S[:, cluster]) == 0:
                    logging.debug("Cluster dropped, one cluster have no positive points anymore, in iteration : %d" % (
                                iteration - 1))
                    logging.debug("Re-initialization of the clustering...")
                    S, cluster_index = self.initialize_clustering(X, y)
                if np.max(S[:, cluster]) < 0.5:
                    logging.debug(
                        "Cluster too far, one cluster have no negative points anymore, in consensus : %d" % (
                                iteration - 1))
                    logging.debug("Re-distribution of this cluster negative weight to 'all'...")
                    S[:, cluster] = 1 / self.n_clusters

            # re-init directions for each clusters
            self.coefficients = {cluster_i: [] for cluster_i in range(self.n_clusters)}
            self.intercepts = {cluster_i: [] for cluster_i in range(self.n_clusters)}
            # run maximization step
            self.maximization_step(X, y, S, iteration)

            # decide the convergence based on the clustering stability
            S_hold = S.copy()
            S, cluster_index = self.expectation_step(X, S, consensus)

            # check the Clustering Stability \w Adjusted Rand Index for stopping criteria
            cluster_consistency = ARI(np.argmax(S, 1), np.argmax(S_hold, 1))

            if cluster_consistency > best_cluster_consistency:
                best_cluster_consistency = cluster_consistency
                self.coefficients[-1] = copy.deepcopy(self.coefficients)
                self.intercepts[-1] = copy.deepcopy(self.intercepts)
                self.orthonormal_basis[-1] = copy.deepcopy(self.orthonormal_basis[consensus])
                self.clustering_method[-1] = copy.deepcopy(self.clustering_method[consensus])
            if cluster_consistency > self.stability_threshold:
                break

        return cluster_index

    def predict_clusters_proba_from_cluster_labels(self, X):
        """Predict positive and negative points clustering probabilities.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.
        Returns
        -------
        S : array-like, shape (n_samples, n_samples)
            Cluster prediction matrix.
        """
        X_clustering_assignments = np.zeros((len(X), self.n_consensus))
        for consensus in range(self.n_consensus):
            X_proj = X @ self.orthonormal_basis[consensus].T
            if self.clustering_method_name in ['k_means', 'full_gaussian_mixture', 'spherical_gaussian_mixture']:
                X_clustering_assignments[:, consensus] = self.clustering_method[consensus].predict(X_proj)
            else:
                return NotImplementedError
        similarity_matrix = compute_similarity_matrix(self.clustering_assignments, clustering_assignments_to_pred=X_clustering_assignments)

        Q = np.zeros((len(X), self.n_clusters))
        y_clusters_train_ = self.cluster_labels_
        for cluster in range(self.n_clusters):
            Q[:, cluster] = np.mean(similarity_matrix[y_clusters_train_ == cluster], 0)
        Q /= np.sum(Q, 1)[:, None]
        return Q

    def clustering_ensembling(self, X, y_polytope):
        """Perform a bagging of the previously obtained clustering and compute new hyperplanes.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors.
        y_polytope : array-like, shape (n_samples,)
            Modified target values.
        Returns
        -------
        None
        """
        # perform consensus clustering
        consensus_cluster_index = compute_spectral_clustering_consensus(self.clustering_assignments, self.n_clusters)
        # save clustering predictions computed by bagging step
        self.cluster_labels_ = consensus_cluster_index

        # update clustering matrix S
        S = self.predict_clusters_proba_from_cluster_labels(X)

        cluster_index = self.run_EM(X, y_polytope, S, consensus_cluster_index, -1)

        # save barycenters and final predictions
        self.cluster_labels_ = cluster_index
        X_proj = X @ self.orthonormal_basis[-1].T
        self.barycenters = [np.mean(X_proj[cluster_index == cluster], 0) for cluster in range(np.max(cluster_index) + 1)]
