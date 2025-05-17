from data import brem_data, brem_sigma, brem_model, brem_data_clean, brem_axis
from data import inter_data, inter_sigma, inter_model
from data import pe_data, pe_data_clean, pe_sigma, pe_model
from midas.parameters import FieldRequest
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
    ),
    name="bremsstrahlung"
)

inter_likelihood = DiagnosticLikelihood(
    diagnostic_model=inter_model,
    likelihood=GaussianLikelihood(
        y_data=inter_data,
        sigma=inter_sigma
    ),
    name="interferometer"
)

pe_likelihood = DiagnosticLikelihood(
    diagnostic_model=pe_model,
    likelihood=GaussianLikelihood(
        y_data=pe_data,
        sigma=pe_sigma
    ),
    name="pressure"
)


from numpy import array, linspace, zeros, exp
field_axis = linspace(0.3, 1.5, 16)
field_models = [
    PiecewiseLinearField(field_name="te", axis=field_axis, axis_name="radius"),
    PiecewiseLinearField(field_name="ne", axis=field_axis, axis_name="radius"),
]

PlasmaState.specify_field_models(field_models)


# ne_smoother = SmoothnessPrior(field="ne", radius=field_axis)
# te_smoother = SmoothnessPrior(field="te", radius=field_axis)

from midas.priors import GaussianProcessPrior
from inference.gp.covariance import SquaredExponential
from inference.gp.mean import ConstantMean

te_gp = GaussianProcessPrior(
    covariance=SquaredExponential(),
    mean=ConstantMean(),
    field_positions=FieldRequest(name="te", coordinates={"radius": field_axis}),
    name="temperature_gp"
)

ne_gp = GaussianProcessPrior(
    covariance=SquaredExponential(),
    mean=ConstantMean(),
    field_positions=FieldRequest(name="ne", coordinates={"radius": field_axis}),
    name="density_gp"
)


posterior = Posterior(components=[brem_likelihood, inter_likelihood, pe_likelihood, ne_gp, te_gp])

print(PlasmaState.parameter_names)
for name, slice in PlasmaState.slices.items():
    print(name, slice)