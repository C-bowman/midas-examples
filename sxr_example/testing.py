from data import brem_data, brem_data_clean, brem_axis
from data import pe_data, pe_data_clean, pe_axis
from data import test_axis, test_ne, test_te, te_profile, ne_profile
from posterior import field_axis, brem_likelihood, pe_likelihood

from midas.state import PlasmaState
from numpy import array, zeros, exp, ptp


te_guess = 500 * exp(-0.5 * ((field_axis - 0.9) / 0.3)**2)
ne_guess = 0.7 * exp(-0.5 * ((field_axis - 0.9) / 0.3)**2)

# te_guess = te_profile(R=field_axis)
# ne_guess = ne_profile(R=field_axis)


initial_guess = PlasmaState.merge_parameters(
    {
        "te_linear_basis": te_guess,
        "ne_linear_basis": ne_guess,
        "background": 3.0,
        "te_cov_hyperpars": [4., -3],
        "ne_cov_hyperpars": [-0.2, -3],
        "te_mean_hyperpars": 0.01,
        "ne_mean_hyperpars": 0.01,
    }
)

from midas.posterior import Posterior
print(">>>>", Posterior.log_probability(initial_guess))
print(Posterior.component_log_probabilities(initial_guess))
# exit()


import matplotlib.pyplot as plt
plt.plot(field_axis, te_guess)
plt.plot(test_axis, test_te)
plt.grid()
plt.show()

lower_bounds = PlasmaState.merge_parameters(
    {
        "te_linear_basis": 1e-6,
        "ne_linear_basis": 1e-6,
        "background": 1e-6,
        "te_cov_hyperpars": [-2, -5],
        "ne_cov_hyperpars": [-2, -5],
        "te_mean_hyperpars": 0.,
        "ne_mean_hyperpars": 0.,
    }
)

upper_bounds = PlasmaState.merge_parameters(
    {
        "te_linear_basis": 1200.,
        "ne_linear_basis": 10.,
        "background": 20.0,
        "te_cov_hyperpars": [6., -2.5],
        "ne_cov_hyperpars": [3., -2.5],
        "te_mean_hyperpars": 1.,
        "ne_mean_hyperpars": 0.05,
    }
)

bounds = [(l, u) for l, u in zip(lower_bounds, upper_bounds)]

from inference.approx import conditional_sample, get_conditionals
from inference.mcmc import EnsembleSampler, Bounds, HamiltonianChain


from scipy.optimize import minimize
result = minimize(
    fun=Posterior.cost,
    x0=initial_guess,
    jac=Posterior.cost_gradient,
    bounds=bounds
)


bounds_class = Bounds(lower=lower_bounds, upper=upper_bounds)

# cond_sample = conditional_sample(
#     posterior=Posterior.log_probability,
#     bounds=bounds,
#     conditioning_point=result.x,
#     n_samples=1000
# )
#
# print(cond_sample.shape)
# p = array([Posterior.log_probability(samp) for samp in cond_sample])
# print(p.shape)

# cond_sample = cond_sample[p.argsort(), :]

x_cond, p_cond = get_conditionals(
    posterior=Posterior.log_probability,
    bounds=bounds,
    conditioning_point=result.x
)

conditional_widths = zeros(len(bounds))
for i in range(len(bounds)):
    extent = x_cond[p_cond[:, i] > p_cond[:, i].max() * 0.1, i]
    conditional_widths[i] = ptp(extent)

chain = HamiltonianChain(
    posterior=Posterior.log_probability,
    start=result.x,
    bounds=bounds_class,
    grad=Posterior.gradient,
    epsilon=0.05,
    inverse_mass=conditional_widths ** 2,
    display_progress=False
)
chain.steps = 30

chain.advance(2500)
chain.plot_diagnostics()
chain.trace_plot()

sample = chain.get_sample(burn=500, thin=2)
print(sample.shape)


params_sample = PlasmaState.split_samples(parameter_samples=sample)

from inference.pdf import sample_hdi
samples_hdi_95 = {
    name: sample_hdi(samples, fraction=0.95) for name, samples in params_sample.items()
}

samples_mean = {name: samples.mean(axis=0) for name, samples in params_sample.items()}


print(PlasmaState.slices)


import matplotlib.pyplot as plt

params = PlasmaState.split_parameters(result.x)

print("\n\n>>>>>>>>>>>>>>>>")
print("initial guess prob", Posterior.log_probability(initial_guess))
print("bfgs solution prob", Posterior.log_probability(result.x))
print("\n\n")

print(params)

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(test_axis, test_ne, label="test values", c="black", ls="dotted")
ax1.plot(field_axis, params["ne_linear_basis"], label="MAP estimate", c="C0", ls="dashed", lw=2)
ax1.plot(field_axis, samples_mean["ne_linear_basis"], label="sample mean", c="C0", lw=2)
ax1.fill_between(field_axis, *samples_hdi_95["ne_linear_basis"], alpha=0.3, color="C0")
ax1.set_ylabel("electron density")
ax1.set_xlabel("major radius")
ax1.grid()
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(test_axis, test_te, color="black", lw=2, ls="dotted", label="test values")
ax2.plot(field_axis, params["te_linear_basis"], color="red", lw=2, ls="dashed", label="MAP estimate")
ax2.plot(field_axis, samples_mean["te_linear_basis"], color="red", lw=2, label="sample mean")
ax2.fill_between(field_axis, *samples_hdi_95["te_linear_basis"], color="red", alpha=0.3)
ax2.set_ylabel("electron temperature (eV)")
ax2.set_xlabel("major radius")
ax2.grid()
ax2.legend()

plt.show()


# print(posterior(ensemble_map))
Posterior.log_probability(result.x)


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