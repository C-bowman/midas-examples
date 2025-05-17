from data import brem_data, brem_data_clean, brem_axis
from data import pe_data, pe_data_clean, pe_axis
from data import test_axis, test_ne, test_te, te_profile, ne_profile
from posterior import posterior, field_axis, brem_likelihood, pe_likelihood

from midas.state import PlasmaState
from numpy import array, zeros, exp


te_guess = 500 * exp(-0.5 * ((field_axis - 0.9) / 0.3)**2)
ne_guess = 0.7 * exp(-0.5 * ((field_axis - 0.9) / 0.3)**2)

# te_guess = te_profile(R=field_axis)
# ne_guess = ne_profile(R=field_axis)


initial_guess = PlasmaState.merge_parameters(
    {
        "te": te_guess,
        "ne": ne_guess,
        "background": 3.0,
        "te_cov_hyperpars": [4., -3],
        "ne_cov_hyperpars": [-0.2, -3],
        "te_mean_hyperpars": 0.01,
        "ne_mean_hyperpars": 0.01,
    }
)


print(">>>>", posterior(initial_guess))
print(posterior.component_log_probabilities(initial_guess))
# exit()


import matplotlib.pyplot as plt
plt.plot(field_axis, te_guess)
plt.plot(test_axis, test_te)
plt.grid()
plt.show()

lower_bounds = PlasmaState.merge_parameters(
    {
        "te": 1e-6,
        "ne": 1e-6,
        "background": 1e-6,
        "te_cov_hyperpars": [-2, -5],
        "ne_cov_hyperpars": [-2, -5],
        "te_mean_hyperpars": 0.,
        "ne_mean_hyperpars": 0.,
    }
)

upper_bounds = PlasmaState.merge_parameters(
    {
        "te": 1200.,
        "ne": 10.,
        "background": 20.0,
        "te_cov_hyperpars": [6., -2.5],
        "ne_cov_hyperpars": [3., -2.5],
        "te_mean_hyperpars": 1.,
        "ne_mean_hyperpars": 0.05,
    }
)

bounds = [(l, u) for l, u in zip(lower_bounds, upper_bounds)]

from inference.approx import conditional_sample
from inference.mcmc import EnsembleSampler, Bounds


from scipy.optimize import minimize
result = minimize(
    fun=posterior.cost,
    x0=initial_guess,
    jac=posterior.cost_gradient,
    bounds=bounds
)


bounds_class = Bounds(lower=lower_bounds, upper=upper_bounds)

cond_sample = conditional_sample(
    posterior=posterior,
    bounds=bounds,
    conditioning_point=result.x,
    n_samples=1000
)

print(cond_sample.shape)
p = array([posterior(samp) for samp in cond_sample])
print(p.shape)

cond_sample = cond_sample[p.argsort(), :]



chain = EnsembleSampler(
    posterior=posterior,
    starting_positions=cond_sample[:, -200:],
    bounds=bounds_class
)

chain.advance(100)
chain.plot_diagnostics()



ensemble_sample = chain.get_sample()
ensemble_probs = chain.get_probabilities()
ensemble_map = ensemble_sample[ensemble_probs.argmax(), :]


print(ensemble_sample.shape)
from scipy.optimize import minimize
# bounds = [(1e-6, None)] * PlasmaState.n_params
# bounds[-2] = (10., None)
# bounds[-1] = (10., None)

print(PlasmaState.slices)

# result = minimize(
#     fun=posterior.cost,
#     x0=initial_guess,
#     jac=posterior.cost_gradient,
#     bounds=bounds
# )
#
# print(result)

import matplotlib.pyplot as plt

params = PlasmaState.split_parameters(ensemble_map)

print("\n\n>>>>>>>>>>>>>>>>")
print("initial guess prob", posterior(initial_guess))
print("bfgs solution prob", posterior(result.x))
print("\n\n")

print(params)


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


# print(posterior(ensemble_map))
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