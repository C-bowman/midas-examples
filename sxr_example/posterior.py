from data import brem_data, brem_sigma, brem_model, brem_data_clean, brem_axis
from data import inter_data, inter_sigma, inter_model
from data import pe_data, pe_data_clean, pe_sigma, pe_model
from prior import SmoothnessPrior

from midas.likelihoods import DiagnosticLikelihood, GaussianLikelihood
from midas.state import PlasmaState
from midas.posterior import Posterior
from midas.fields import PiecewiseLinearField


brem_likelihood = DiagnosticLikelihood(
    diagnostic_model=brem_model,
    likelihood=GaussianLikelihood(
        y_data=brem_data,
        sigma=brem_sigma
    )
)

inter_likelihood = DiagnosticLikelihood(
    diagnostic_model=inter_model,
    likelihood=GaussianLikelihood(
        y_data=inter_data,
        sigma=inter_sigma
    )
)

pe_likelihood = DiagnosticLikelihood(
    diagnostic_model=pe_model,
    likelihood=GaussianLikelihood(
        y_data=pe_data,
        sigma=pe_sigma
    )
)


from numpy import array, linspace, zeros, exp
field_axis = linspace(0.3, 1.5, 30)
field_models = [
    PiecewiseLinearField(field_name="te", axis=field_axis, axis_name="radius"),
    PiecewiseLinearField(field_name="ne", axis=field_axis, axis_name="radius"),
]

PlasmaState.specify_field_models(field_models)


ne_smoother = SmoothnessPrior(field="ne", radius=field_axis)
te_smoother = SmoothnessPrior(field="te", radius=field_axis)


# posterior = Posterior(components=[brem_likelihood, inter_likelihood, ne_smoother, te_smoother])
posterior = Posterior(components=[brem_likelihood, inter_likelihood, pe_likelihood, ne_smoother, te_smoother])
# posterior = Posterior(components=[brem_likelihood, inter_likelihood, pe_likelihood])
