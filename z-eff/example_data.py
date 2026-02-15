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


"""
Generate 'true' profiles of the fields we want to infer
"""
measurement_radius = linspace(0.9, 1.35, 50)

true_te_profile = mtanh(
    radius=measurement_radius,
    theta=[1.38, 120., 0.04, 100., 5.]
) + 400 * exp(-0.5 * ((measurement_radius - 0.9) / 0.3)**4)


true_ne_profile = mtanh(
    radius=measurement_radius,
    theta=[1.385, 5e19, 0.03, -2e19, 1e18]
) + 2e19 * exp(-0.5 * ((measurement_radius - 0.87) / 0.15)**2)

true_z_eff_profile = 2. + logistic(x=measurement_radius, c=1.5, w=0.035) + exp(-0.5 * ((measurement_radius - 0.9) / 0.15) ** 2)


"""
Set up the model for the bremsstrahlung and generate diagnostic predictions
"""
brem_model = BremsstrahlungModel(
    radius=measurement_radius,
    wavelength=569e-9,
)

brem_predictions = brem_model.predictions(
    te=true_te_profile,
    ne=true_ne_profile,
    z_eff=true_z_eff_profile
)


"""
Add sampled noise to the diagnostic predictions to generate synthetic data
"""
rng = default_rng(236)

brem_sigma = brem_predictions * 0.05 + brem_predictions.max()*0.01
brem_measurements = brem_predictions + rng.normal(scale=brem_sigma)

te_sigma = true_te_profile * 0.05 + 1.0
te_measurements = true_te_profile + rng.normal(scale=te_sigma)

ne_sigma = true_ne_profile * 0.04 + 0.5e18
ne_measurements = true_ne_profile + rng.normal(scale=ne_sigma)



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.plot(measurement_radius, true_te_profile)
    ax1.plot(measurement_radius, te_measurements, ".")
    ax1.set_ylim([0, None])
    ax1.grid()

    ax2.plot(measurement_radius, true_ne_profile)
    ax2.plot(measurement_radius, ne_measurements, ".")
    ax2.set_ylim([0, None])
    ax2.grid()

    ax3.plot(measurement_radius, brem_predictions)
    ax3.plot(measurement_radius, brem_measurements, ".")
    ax3.set_ylim([0, None])
    ax3.grid()

    fig.tight_layout()
    plt.show()