from midas import FieldRequest
from midas.likelihoods import DiagnosticLikelihood
from midas.likelihoods import GaussianLikelihood
from diagnostics import BremsstrahlungModel, ThomsonModel

from synthetic_data import measurement_radius, brem_measurements, brem_sigma
from synthetic_data import te_measurements, te_sigma, ne_measurements, ne_sigma
from numpy import full, log


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
        wavelength=569e-9,
    )

    brem_diagnostic = DiagnosticLikelihood(
        likelihood=brem_likelihood,
        diagnostic_model=brem_model,
        name="brem_diagnostic"
    )


    """
    Build the Te profile diagnostic
    """
    te_likelihood = GaussianLikelihood(
        y_data=te_measurements,
        sigma=te_sigma,
    )

    te_model = ThomsonModel(
        radius=measurement_radius,
        field="te"
    )

    te_diagnostic = DiagnosticLikelihood(
        likelihood=te_likelihood,
        diagnostic_model=te_model,
        name="te_diagnostic"
    )


    """
    Build the ne profile diagnostic
    """
    ne_likelihood = GaussianLikelihood(
        y_data=ne_measurements,
        sigma=ne_sigma,
    )

    ne_model = ThomsonModel(
        radius=measurement_radius,
        field="ne"
    )

    ne_diagnostic = DiagnosticLikelihood(
        likelihood=ne_likelihood,
        diagnostic_model=ne_model,
        name="ne_diagnostic"
    )



    """
    Build the field models
    """
    from numpy import linspace
    from midas.models.fields import ExSplineField, CubicSplineField

    ts_knots = linspace(0.9, 1.35, 6)
    ne_field = ExSplineField(
        field_name="ne",
        axis_name="radius",
        axis=ts_knots,
    )

    te_field = ExSplineField(
        field_name="te",
        axis_name="radius",
        axis=ts_knots,
    )

    z_eff_knots = linspace(0.9, 1.35, 5)
    z_eff_field = CubicSplineField(
        field_name="z_eff",
        axis=z_eff_knots,
        axis_name="radius"
    )

    from midas.priors import SoftLimitPrior
    from midas.operators import derivative_operator
    te_request = FieldRequest(name="te", coordinates={"radius": measurement_radius})
    operator = derivative_operator(measurement_radius, order=1)

    te_monotonicity_prior = SoftLimitPrior(
        name="te_monotonicity_prior",
        field_request=te_request,
        upper_limit=0.0,
        sigma=30.,
        operator=operator
    )



    from midas import PlasmaState, Parameters

    PlasmaState.build_posterior(
        diagnostics=[brem_diagnostic, te_diagnostic, ne_diagnostic],
        priors=[te_monotonicity_prior],
        field_models=[te_field, ne_field, z_eff_field]
    )



    """
    Find MAP estimate by maximising the posterior log-probability
    """
    initial_guess_dict = {
        'ln_ne_bspline_basis': full(ts_knots.size, fill_value=log(5e19)),
        'ln_te_bspline_basis': full(ts_knots.size, fill_value=log(60.)),
        'z_eff_cubic_spline': full(z_eff_knots.size, fill_value=1.2),
    }
    initial_guess_array = PlasmaState.merge_parameters(initial_guess_dict)

    bounds_dict = {
        'ln_ne_bspline_basis': (log(1e17), log(1e21)),
        'ln_te_bspline_basis': (log(2), log(1000)),
        'z_eff_cubic_spline': (1.0, 5.0),
    }
    bounds = PlasmaState.build_bounds(bounds_dict)


    from midas import posterior
    from scipy.optimize import minimize, approx_fprime

    numgrad = approx_fprime(f=posterior.log_probability, xk=initial_guess_array)
    print(posterior.gradient(initial_guess_array) / numgrad)

    opt_result = minimize(
        fun=posterior.cost,
        x0=initial_guess_array,
        method='L-BFGS-B',
        bounds=bounds,
        jac=posterior.cost_gradient,
    )




    """
    Plot the data predictions based on the MAP estimate
    """
    map_predictions = posterior.get_model_predictions(opt_result.x)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    data_style = dict(marker="o", linestyle="none", markerfacecolor="none", color="black")

    ax1.plot(measurement_radius, map_predictions["ne_diagnostic"], color="blue", lw=2)
    ax1.errorbar(measurement_radius, ne_measurements, yerr=ne_sigma, **data_style)
    ax1.set_xlabel("Radius (m)")
    ax1.set_ylabel("Electron density (m^-3)")
    ax1.grid()

    ax2.plot(measurement_radius, map_predictions["te_diagnostic"], color="red", lw=2)
    ax2.errorbar(measurement_radius, te_measurements, yerr=te_sigma, **data_style)
    ax2.set_xlabel("Radius (m)")
    ax2.set_ylabel("Electron temperature (eV)")
    ax2.grid()

    ax3.plot(measurement_radius, map_predictions["brem_diagnostic"], c="green", lw=2)
    ax3.errorbar(measurement_radius, brem_measurements, yerr=brem_sigma, **data_style)
    ax3.set_xlabel("Radius (m)")
    ax3.set_ylabel("Z-effective")
    ax3.grid()

    fig.tight_layout()
    plt.show()


    """
    Use MCMC to sample from the posterior distribution
    """
    from inference.mcmc import HamiltonianChain
    from inference.approx import conditional_moments

    _, conditional_variance = conditional_moments(
        posterior=posterior.log_probability,
        conditioning_point=opt_result.x,
        bounds=[b for b in bounds],
    )

    chain = HamiltonianChain(
        posterior=posterior.log_probability,
        grad=posterior.gradient,
        start=opt_result.x,
        inverse_mass=conditional_variance,
        bounds=(bounds[:, 0], bounds[:, 1]),
        epsilon=0.25
    )

    chain.advance(5000)
    chain.trace_plot()
    chain.plot_diagnostics()
    sample = chain.get_sample(burn=1000, thin=1)



    test = posterior.sample_model_predictions(sample)
    from inference.plotting import hdi_plot



    from midas import FieldRequest
    profile_axis = linspace(0.9, 1.35, 128)

    te_profiles = posterior.sample_field_values(
        parameter_samples=sample,
        field_request=FieldRequest("te", coordinates={"radius": profile_axis}),
    )

    ne_profiles = posterior.sample_field_values(
        parameter_samples=sample,
        field_request=FieldRequest("ne", coordinates={"radius": profile_axis}),
    )

    z_eff_profiles = posterior.sample_field_values(
        parameter_samples=sample,
        field_request=FieldRequest("z_eff", coordinates={"radius": profile_axis}),
    )




    from synthetic_data import z_eff_profile, te_profile, ne_profile

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    ax1.plot(measurement_radius, te_profile, lw=2, color="black", ls="dashed", label=r"true $T_e$")
    hdi_plot(profile_axis, te_profiles, axis=ax1, color="red")
    ax1.set_xlabel("Radius (m)")
    ax1.set_ylabel("electron temperature (eV)")
    ax1.set_xlim([0.9, 1.35])
    ax1.grid()
    ax1.legend()


    ax2.plot(measurement_radius, ne_profile, lw=2, color="black", ls="dashed", label=r"true $n_e$")
    hdi_plot(profile_axis, ne_profiles, axis=ax2, color="C0")
    ax2.set_xlabel("Radius (m)")
    ax2.set_ylabel("electron density (m^-3)")
    ax2.set_xlim([0.9, 1.35])
    ax2.grid()
    ax2.legend()


    ax3.plot(measurement_radius, z_eff_profile, lw=2, color="black", ls="dashed", label="true Z-eff")
    hdi_plot(profile_axis, z_eff_profiles, axis=ax3, color="green", intervals=[0.9, 0.5])
    ax3.set_xlabel("Radius (m)")
    ax3.set_ylabel("Z-effective")
    ax3.set_xlim([0.9, 1.35])
    ax3.grid()
    ax3.legend()

    plt.tight_layout()
    plt.show()

















