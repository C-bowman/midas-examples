from midas.parameters import ParameterVector, FieldRequest
from midas.priors import BasePrior
from numpy import array, ndarray, ones, zeros, log
from scipy.special import factorial
from scipy.linalg import solve


def derivative_operator(radius: ndarray, order=1):
    A = zeros([radius.size, radius.size])
    for i in range(1, radius.size - 1):
        A[i, i - 1 : i + 2] = findiff_coeffs(
            radius[i - 1 : i + 2] - radius[i], order=order
        )
    n = 2 + order
    A[0, :n] = findiff_coeffs(radius[:n] - radius[0], order=order)
    A[-1, -n:] = findiff_coeffs(radius[-n:] - radius[-1], order=order)
    return A


def findiff_coeffs(points: ndarray, order=1):
    # check validity of inputs
    if type(points) is not ndarray:
        points = array(points)
    n = len(points)
    if n <= order:
        raise ValueError(
            "The order of the derivative must be less than the number of points"
        )
    # build the linear system
    b = zeros(n)
    b[order] = factorial(order)
    A = ones([n, n])
    for i in range(1, n):
        A[i, :] = points**i
    # return the solution
    return solve(A, b)


class SmoothnessPrior(BasePrior):
    def __init__(self, field: str, radius: ndarray):
        self.field_name = field
        self.param_name = f"{field}_smoothness"
        self.parameters = [ParameterVector(name=self.param_name, size=1)]
        self.field_requests = [FieldRequest(name=self.field_name, coordinates={"radius": radius})]
        self.operator = derivative_operator(radius, order=2)
        self.S = self.operator.T @ self.operator

    def probability(self, **kwargs) -> float:
        field = kwargs[self.field_name]
        sigma = kwargs[self.param_name]
        df = self.operator @ field
        return -0.5*(df**2).sum() / sigma**2 - log(sigma)

    def gradients(self, **kwargs) -> dict[str, ndarray]:
        field = kwargs[self.field_name]
        sigma = kwargs[self.param_name]
        dPdf = -(self.S @ field) / sigma**2
        dPds = -((field * dPdf).sum() - 1) / sigma
        return {
            self.field_name: dPdf,
            self.param_name: dPds
        }
