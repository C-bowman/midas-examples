from midas.likelihoods import DiagnosticLikelihood
from midas.likelihoods import GaussianLikelihood
from diagnostics import BremsstrahlungModel, ThomsonTempModel, ThomsonDensityModel
from synthetic_data import measurement_radius, brem_measurements, brem_sigma


if __name__ == "__main__":

    """
    Build the Bremsstrahlung diagnostic
    """

    brem_likelihood = GaussianLikelihood(
        y_data=brem_measurements,
        sigma=brem_sigma,
    )

    brem_model = BremsstrahlungModel(
        radius=measurement_radius,
        wavelength=569.0,
    )

    brem_diagnostic = DiagnosticLikelihood(
        likelihood=brem_likelihood,
        diagnostic_model=brem_model
    )


    """
    Build the field models
    """
    from numpy import linspace
    from fields import CubicProfile


    spline_knots = linspace(1.25, 1.5, 6)
    ne_field = CubicProfile(
        xknots=spline_knots,
        name="ne",
        axis_name="radius"
    )

    te_field = CubicProfile(
        xknots=spline_knots,
        name="ne",
        axis_name="radius"
    )

    z_eff_field = PiecewiseLinearField(
        field_name="z_eff",
        axis=...,
        axis_name="radius"
    )







    from midas import PlasmaState
    PlasmaState.build_posterior(
        diagnostics=[brem_diagnostic],
        priors=[],
        field_models=[te_field, ne_field, z_eff_field]
    )

    from midas import posterior


























