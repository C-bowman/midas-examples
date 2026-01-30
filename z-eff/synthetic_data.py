from numpy import linspace, exp, ndarray
from numpy.random import default_rng
from diagnostics import BremsstrahlungModel


def logistic(x, w, c):
    z = (x - c) / w
    return 1 / (1 + exp(-z))


def mtanh(radius: ndarray, theta: ndarray, drn=-1.0) -> ndarray:
    """
    Calculates the prediction of the ``mtanh`` model.
    See the documentation for ``mtanh`` for details of the model itself.

    :param radius: \
        Radius values at which the prediction is evaluated.

    :param theta: \
        The model parameters as an array.

    :return: \
        The predicted profile at the given radius values.
    """
    R0, h, w, a, b = theta
    sigma = 0.25 * w
    z = (radius - R0) * (drn / sigma)
    G = h - b + (a * sigma) * z
    iL = 1 + exp(-z)
    return (G / iL) + b



measurement_radius = linspace(1.25, 1.5, 26)
te_profile = mtanh(
    radius=measurement_radius,
    theta=[1.38, 120., 0.03, 1000., 2.]
)

ne_profile = mtanh(
    radius=measurement_radius,
    theta=[1.385, 5e19, 0.03, 100., 7e17]
)

z_eff_profile = 2. + logistic(x=measurement_radius, c=1.5, w=0.035)


measurement_radius = linspace(1.25, 1.5, 26)

brem_model = BremsstrahlungModel(
    radius=measurement_radius,
    wavelength=569.0,
)

brem_predictions = brem_model.predictions(
    te=te_profile,
    ne=ne_profile,
    z_eff=z_eff_profile
)

rng = default_rng()

brem_sigma = brem_predictions * 0.05 + brem_predictions.max()*0.01
brem_measurements = brem_predictions + rng.normal(scale=brem_sigma)


te_sigma = te_profile * 0.05 + te_profile.max()*0.01
te_measurements = te_profile + rng.normal(scale=te_sigma)

ne_sigma = ne_profile * 0.05 + ne_profile.max()*0.01
ne_measurements = ne_profile + rng.normal(scale=ne_sigma)