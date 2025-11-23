from numpy import array, linspace, exp, zeros
from numpy.random import default_rng
from models import BremstrahlModel, PressureModel, InterferometerModel
rng = default_rng(137)


def logistic(x, c, w):
    z = (x - c) / w
    return 1 / (1 + exp(-z))

def ne_profile(R):
    return logistic(R, 0.5, 0.05) * (1 - logistic(R, 1.3, 0.05))

def te_profile(R):
    z = (R - 0.9) / 0.2
    return 800 * exp(-0.5 * z ** 2)


# define test profiles for electron temperature and density
R_min, R_max = 0.3, 1.5


# generate synthetic data for the interferometer
inter_axis = linspace(R_min, R_max, 64)
inter_model = InterferometerModel(radius=inter_axis)
inter_data_clean = inter_model.predictions(ne=ne_profile(inter_axis))
inter_sigma = array([2.])
inter_data = inter_data_clean + rng.normal(scale=inter_sigma)

# generate synthetic data for the SXR emissivity
brem_axis = linspace(R_min, R_max, 28)
brem_model = BremstrahlModel(radius=brem_axis)
brem_data_clean = brem_model.predictions(
    te=te_profile(brem_axis),
    ne=ne_profile(brem_axis),
    background=5.
)
brem_sigma = zeros(brem_data_clean.size) + 2
brem_data = brem_data_clean + rng.normal(size=brem_data_clean.size, scale=brem_sigma)


# generate synthetic data for the pressure profile
pe_axis = linspace(R_min, R_max, 64)
pe_model = PressureModel(radius=pe_axis)
pe_data_clean = pe_model.predictions(
    ne=ne_profile(R=pe_axis),
    te=te_profile(R=pe_axis),
)
pe_sigma = pe_data_clean * 0.05 + pe_data_clean.max() * 0.02
pe_data = abs(pe_data_clean + rng.normal(size=pe_data_clean.size, scale=pe_sigma))


test_axis = linspace(R_min, R_max, 64)
test_te = te_profile(R=test_axis)
test_ne = ne_profile(R=test_axis)