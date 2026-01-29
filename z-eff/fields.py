from numpy import ndarray
from midas import Parameters, FieldRequest
from midas.models import FieldModel
from typing import override
from scipy.interpolate import CubicSpline


class CubicSplineField(FieldModel):
    def __init__(self, xknots: ndarray, name: str, axis_name: str):
        self.xknots = xknots
        self.n_params = xknots.size
        self.field_name = name
        self.axis_name = axis_name
        self.parameter_name = f"{self.field_name}_cubic_spline"
        self.parameters = Parameters(
            (self.parameter_name, self.n_params)
        )

    @override
    def get_values(self, parameters: dict[str, ndarray], field: FieldRequest) -> ndarray:
        x_requested = field.coordinates[self.axis_name]
        y_knots = parameters[self.parameter_name]

        spline = CubicSpline(self.xknots, y_knots, bc_type="clamped")
        values = spline(x_requested)
        return values.clip(min=0.0)

    @override
    def get_values_and_jacobian(
        self, parameters: dict[str, ndarray], field: FieldRequest
    ):
        raise NotImplementedError()
