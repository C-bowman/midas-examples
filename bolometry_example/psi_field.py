from midas.fields import FieldModel
from numpy import array, ndarray, zeros, diff
from midas.parameters import FieldRequest, ParameterVector
from abc import ABC, abstractmethod

coordinates = dict[str, ndarray]


class CoordinateTransform(ABC):
    inputs: set[str]
    outputs: set[str]

    @abstractmethod
    def __call__(self, input_coords: coordinates) -> coordinates:
        pass


class PsiTransform(CoordinateTransform):
    def __init__(self, spline):
        self.inputs = {"R", "z"}
        self.outputs = {"psi"}
        self.spline = spline

    def __call__(self, coords: coordinates) -> coordinates:
        points = array([coords["R"], coords["z"]]).T
        psi = self.spline(points)
        return {"psi": psi}


class PiecewiseFluxProfile(FieldModel):
    def __init__(
        self,
        field_name: str,
        axis: ndarray,
        axis_name: str,
        psi_transform: CoordinateTransform
    ):
        assert axis.ndim == 1
        assert axis.size > 1
        assert (diff(axis) > 0.0).all()
        assert axis.min() >= 0.
        assert axis.max() <= 1.

        self.name = field_name
        self.n_params = axis.size
        self.axis = axis
        self.axis_name = axis_name
        self.matrix_cache = {}
        self.param_name = f"{field_name}_flux_profile_basis"
        self.parameters = [ParameterVector(name=self.param_name, size=self.n_params)]
        self.psi_transform = psi_transform

    def get_basis(self, field: FieldRequest) -> ndarray:
        if field in self.matrix_cache:
            A = self.matrix_cache[field]

        else:
            psi_norm = self.psi_transform(field.coordinates)["psi"]
            A = self.build_linear_basis(
                x=psi_norm, knots=self.axis
            )
            self.matrix_cache[field] = A
        return A

    def get_values(
        self, parameters: dict[str, ndarray], field: FieldRequest
    ) -> ndarray:
        basis = self.get_basis(field)
        return basis @ parameters[self.param_name]

    def get_values_and_jacobian(
        self, parameters: dict[str, ndarray], field: FieldRequest
    ) -> tuple[ndarray, dict[str, ndarray]]:
        basis = self.get_basis(field)
        return basis @ parameters[self.param_name], {self.param_name: basis}

    @staticmethod
    def build_linear_basis(x: ndarray, knots: ndarray) -> ndarray:
        basis = zeros([x.size, knots.size])
        for i in range(knots.size - 1):
            k = ((x >= knots[i]) & (x <= knots[i + 1])).nonzero()
            basis[k, i + 1] = (x[k] - knots[i]) / (knots[i + 1] - knots[i])
            basis[k, i] = 1 - basis[k, i + 1]
        return basis