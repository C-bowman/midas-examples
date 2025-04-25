from data import brem_data, brem_data_clean, brem_axis
from data import pe_data, pe_data_clean, pe_axis
from data import test_axis, test_ne, test_te
from posterior import posterior, field_axis, brem_likelihood, pe_likelihood

from midas.state import PlasmaState
from numpy import array, zeros, exp


initial_guess = PlasmaState.merge_parameters(
    {
        "te": 100.,
        "ne": 0.5,
        "background": 3.0,
        "ne_smoothness": 10.,
        "te_smoothness": 10.,
    }
)

from scipy.optimize import minimize
bounds = [(1e-6, None)] * PlasmaState.n_params
# bounds[-2] = (10., None)
# bounds[-1] = (10., None)

print(PlasmaState.slices)

result = minimize(
    fun=posterior.cost,
    x0=initial_guess,
    jac=posterior.cost_gradient,
    bounds=bounds
)

import matplotlib.pyplot as plt

params = PlasmaState.split_parameters(result.x)

plt.plot(test_axis, test_ne, label="test values")
plt.plot(field_axis, params["ne"], label="inferred")
plt.ylabel("electron density")
plt.xlabel("major radius")
plt.grid()
plt.legend()
plt.show()

plt.plot(test_axis, test_te, label="test values")
plt.plot(field_axis, params["te"], label="inferred")
plt.ylabel("electron temperature (eV)")
plt.xlabel("major radius")
plt.grid()
plt.legend()
plt.show()


posterior(result.x)

brem_fit = brem_likelihood.get_predictions()

plt.plot(brem_axis, brem_data, "o", label="synthetic measurements", markerfacecolor="none", markeredgewidth=2)
plt.plot(brem_axis, brem_data_clean, ".--", alpha=0.5, label="test values")
plt.plot(brem_axis, brem_fit, c="red", lw=2, label="inference predictions")
plt.grid()
plt.legend()
plt.show()

pe_fit = pe_likelihood.get_predictions()
plt.plot(pe_axis, pe_data, "o", label="synthetic measurements", markerfacecolor="none", markeredgewidth=2)
plt.plot(pe_axis, pe_data_clean, ".--", alpha=0.5, label="test values")
plt.plot(pe_axis, pe_fit, c="red", lw=2, label="inference predictions")
plt.grid()
plt.legend()
plt.show()