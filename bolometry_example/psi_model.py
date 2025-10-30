from numpy import linspace, sin, cos, exp, log, sqrt, meshgrid, atan2, zeros
import matplotlib.pyplot as plt

from scipy.interpolate import CloughTocher2DInterpolator
from tokamesh.tokamaks import mastu_boundary


class ParametrisedPsi:
    def __init__(
        self,
        r_min: float,
        axis: tuple[float, float],
        elongation=1.75,
        triangularity=0.6
    ):
        self.axis = axis
        self.r_min = r_min
        self.kappa_max = elongation
        self.delta_max = triangularity
        self.psi_max = 1.05

        n = 128
        x = linspace(-2, 3, n)
        y = linspace(-2.5, 2.5, n)
        x_mesh, y_mesh = meshgrid(x, y, indexing="ij")
        theta = atan2(y_mesh, x_mesh)
        rho = sqrt(x_mesh ** 2 + y_mesh ** 2)

        R, z = self.flux_contour(rho, theta)
        psi_norm = self.psi(rho).flatten()

        coords = zeros([R.size, 2])
        coords[:, 0] = R.flatten()
        coords[:, 1] = z.flatten()

        self.spline = CloughTocher2DInterpolator(
            points=coords,
            values=psi_norm,
        )

    def psi(self, rho):
        sigma = sqrt(-0.5 / log((self.psi_max - 1) / self.psi_max))
        return (1 - exp(-0.5*(rho / sigma)**2)) * self.psi_max

    def shape_params(self, rho):
        scale = 0.3
        gaussian_ramp = 1 - exp(-0.5*(rho / scale)**2)
        delta = gaussian_ramp * self.delta_max
        kappa = 1 + gaussian_ramp * (self.kappa_max - 1)
        return delta, kappa

    def flux_contour(self, rho, theta):
        R0, z0 = self.axis
        delta, kappa = self.shape_params(rho)
        R = rho * self.r_min * cos(theta + delta * sin(theta)) + R0
        z = rho * self.r_min * kappa * sin(theta) + z0
        return R, z



if __name__ == "__main__":


    psi = ParametrisedPsi(
        r_min=0.5,
        axis=(0.85, 0.01),
        elongation=1.75,
        triangularity=0.6,
    )


    R_axis = linspace(0.25, 1.8, 32)
    z_axis = linspace(-1.8, 1.8, 32)
    R_mesh, z_mesh = meshgrid(R_axis, z_axis, indexing="ij")


    points = zeros([R_mesh.size, 2])
    points[:, 0] = R_mesh.flatten()
    points[:, 1] = z_mesh.flatten()


    psi_interp = psi.spline(points)
    psi_interp.resize(R_mesh.shape)
    print(psi_interp.min(), psi_interp.max())

    plt.contourf(R_axis, z_axis, psi_interp.T, 32)
    plt.contour(R_axis, z_axis, psi_interp.T, levels=[1.0], colors=["red"])
    plt.plot(*mastu_boundary(), lw=2, color="black")
    plt.axis("equal")
    plt.show()
