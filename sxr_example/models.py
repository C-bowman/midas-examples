from numpy import ndarray, sqrt, ones, fill_diagonal, zeros
from midas.models import DiagnosticModel
from midas.parameters import ParameterVector, FieldRequest


class BremstrahlModel(DiagnosticModel):
    def __init__(self, radius: ndarray):
        self.parameters = [ParameterVector(name="background", size=1)]
        self.field_requests = [
            FieldRequest(name="te", coordinates={"radius": radius}),
            FieldRequest(name="ne", coordinates={"radius": radius}),
        ]

    def predictions(self, te: ndarray, ne: ndarray, background: float):
        return sqrt(te) * ne**2 + background

    def predictions_and_jacobians(self, te: ndarray, ne: ndarray, background: float):
        predictions = sqrt(te) * ne**2 + background
        te_jac = zeros([te.size, te.size])
        ne_jac = zeros([te.size, te.size])
        bg_jac = zeros((te.size, 1)) + 1.0
        fill_diagonal(te_jac, 0.5*ne**2 / sqrt(te))
        fill_diagonal(ne_jac, 2 * sqrt(te) * ne)
        jacobians = {
            "te": te_jac,
            "ne": ne_jac,
            "background": bg_jac
        }
        return predictions, jacobians


class PressureModel(DiagnosticModel):
    def __init__(self, radius: ndarray):
        self.parameters = []
        self.field_requests = [
            FieldRequest(name="te", coordinates={"radius": radius}),
            FieldRequest(name="ne", coordinates={"radius": radius}),
        ]

    def predictions(self, te: ndarray, ne: ndarray):
        return te * ne

    def predictions_and_jacobians(self, te: ndarray, ne: ndarray):
        predictions = te * ne
        te_jac = zeros([te.size, te.size])
        ne_jac = zeros([te.size, te.size])
        fill_diagonal(te_jac, ne)
        fill_diagonal(ne_jac, te)
        jacobians = {
            "te": te_jac,
            "ne": ne_jac,
        }
        return predictions, jacobians


class InterferometerModel(DiagnosticModel):
    def __init__(self, radius: ndarray):
        self.parameters = []
        self.field_requests = [
            FieldRequest(name="ne", coordinates={"radius": radius})
        ]

    def predictions(self, ne: ndarray):
        return ne.sum()

    def predictions_and_jacobians(self, ne: ndarray):
        predictions = ne.sum()
        jacobians = {
            "ne": ones(shape=(1, ne.size)),
        }
        return predictions, jacobians