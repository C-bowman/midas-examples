from abc import ABC, abstractmethod
from numpy import sqrt, diagonal, eye, exp, log, pi, maximum
from numpy import array, ndarray, linspace, zeros, empty
from numpy.linalg import cholesky, LinAlgError
from scipy.linalg import solve_triangular
from warnings import warn
from inference.covariance import CovarianceFunction
from inference.mean import MeanFunction

from midas.parameters import ParameterVector, FieldRequest


class GaussianProcessPrior:
    def __init__(
        self,
        covariance: CovarianceFunction,
        mean: MeanFunction,
        field: str,
        coordinates: dict[str, ndarray]
    ):
        self.cov = covariance
        self.mean = mean
        self.field = field
        self.I = eye(PlasmaState.radius.size)

        spatial_data = PlasmaState.radius.reshape([PlasmaState.radius.size, 1])
        self.cov.pass_spatial_data(spatial_data)
        self.mean.pass_spatial_data(spatial_data)

        self.field_requests = [
            FieldRequest()
        ]

        self.cov_tag = f"{self.field}_cov_hyperpars"
        self.mean_tag = f"{self.field}_mean_hyperpars"
        self.parameters = [
            ParameterVector(tag=self.field, size=PlasmaState.radius.size),
            ParameterVector(tag=self.cov_tag, size=self.cov.n_params),
            ParameterVector(tag=self.mean_tag, size=self.mean.n_params),
        ]

    def probability(self, **kwargs):
        K = self.cov.build_covariance(PlasmaState.get(self.cov_tag))
        mu = self.mean.build_mean(PlasmaState.get(self.mean_tag))
        try:  # protection against singular matrix error crash
            L = cholesky(K)
            v = solve_triangular(L, PlasmaState.get(self.field) - mu, lower=True)
            return -0.5 * (v @ v) - log(diagonal(L)).sum()
        except LinAlgError:
            warn("Cholesky decomposition failure in marginal_likelihood")
            return -1e50

    def gradient(self):
        K, grad_K = self.cov.covariance_and_gradients(PlasmaState.get(self.cov_tag))
        mu, grad_mu = self.mean.mean_and_gradients(PlasmaState.get(self.mean_tag))
        # get the cholesky decomposition
        L = cholesky(K)
        iK = solve_triangular(L, self.I, lower=True)
        iK = iK.T @ iK
        # from scipy.linalg import inv
        # print(">> INVERSION CHECK", ((iK @ K) - self.I).max(), ((inv(K) @ K) - self.I).max())
        # iK = inv(K)
        # calculate the log-marginal likelihood
        dy = PlasmaState.get(self.field) - mu
        alpha = iK @ dy
        # LML = -0.5 * dot((self.y - mu).T, alpha) - log(diagonal(L)).sum()
        # calculate the mean parameter gradients
        grad = zeros(PlasmaState.n_params)
        grad[PlasmaState.slices[self.mean_tag]] = array(
            [(alpha * dmu).sum() for dmu in grad_mu]
        )
        # calculate the covariance parameter gradients
        Q = alpha[:, None] * alpha[None, :] - iK
        grad[PlasmaState.slices[self.cov_tag]] = array(
            [0.5 * (Q * dK.T).sum() for dK in grad_K]
        )
