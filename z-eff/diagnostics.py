from numpy import ndarray, exp, sqrt, log, eye, diagflat
from midas.models import DiagnosticModel
from midas import Fields, Parameters, FieldRequest


class ThomsonModel(DiagnosticModel):
    def __init__(self, radius: ndarray, field: str):
        self.field_name = field
        self.fields = Fields(
            FieldRequest(name=field, coordinates={"radius": radius})
        )
        self.parameters = Parameters()
        self.jacobian = eye(radius.size)

    def predictions(self, **fields) -> ndarray:
        return fields[self.field_name]

    def predictions_and_jacobians(
            self, **fields
    ) -> tuple[ndarray, dict[str, ndarray]]:
        return fields[self.field_name], {self.field_name: self.jacobian}


class BremsstrahlungModel(DiagnosticModel):
    def __init__(self, radius: ndarray, wavelength: float):
        self.wavelength = wavelength

        self.fields = Fields(
            FieldRequest(name="te", coordinates={"radius": radius}),
            FieldRequest(name="ne", coordinates={"radius": radius}),
            FieldRequest(name="z_eff", coordinates={"radius": radius}),
        )
        self.parameters = Parameters()

    def predictions(self, te: ndarray, ne: ndarray, z_eff: ndarray) -> ndarray:
        # call model / return predictions
        return bremsstrahlung_model(
            Te=te,
            Ne=ne,
            zeff=z_eff,
            wavelength=self.wavelength,
        )

    def predictions_and_jacobians(
            self, te: ndarray, ne: ndarray, z_eff: ndarray
    ) -> tuple[ndarray, dict[str, ndarray]]:
        return bremsstrahlung_jacobian(
            Te=te,
            Ne=ne,
            zeff=z_eff,
            wavelength=self.wavelength,
        )


def zeff_bremsstrahlung(
        Te: ndarray,
        Ne: ndarray,
        wavelength: float,
        zeff: ndarray = None,
        bremsstrahlung: ndarray = None,
        gaunt_approx="callahan",
) -> ndarray:
    """
    Calculate Bremsstrahlung <-> Zeff using the formulas from:
    "Zeff Measurement and Analysis of Reheat-Mode Discharges in W7-X"
    S. Morita , IPP III / 199 , 1994
    https://core.ac.uk/download/pdf/210800245.pdf

    Parameters
    ----------
    Te
        electron temperature (eV) for calculation
    Ne
        electron density in (m^-3)
    wavelength
        filter central wavelength (nm)
    zeff
        effective charge
    bremsstrahlung
        Bremsstrahlung emission in units [W / m^3 / nm]
    gaunt_approx
        approximation for free-free gaunt factors:
            "callahan" see KJ Callahan 2019 JINST 14 C10002
            "carson" see TR Carson 1988 Atron. Atrophys. 1889, 319324

    Returns
    -------
    Zeff
        or
    Bremsstrahlung [W / m^3 / nm]

    N.B. Comparison with experimental Bremsstrahlung spectrally-resolved
    measurement requires LOS-integral & multiplication by the system's
    Etendue / 4 / np.pi. For filtered diode, the result should also be
    multiplied by the transmission curve and integrated over wavelength.
    """

    assert (zeff is not None) == (bremsstrahlung is None)

    wavelength_ang = wavelength * 10  # nm to Angstrom

    gaunt_funct = {
        "callahan": 1.35 * Te ** 0.15,
        "carson": -4.7499 + 0.5513 * log(Te * wavelength_ang),
    }
    gaunt = gaunt_funct[gaunt_approx]

    Ne_cm = Ne * 1e-6  # m^-3 to cm^-3

    reconvert_units = 1e6 * 10  # convert back to (nm) and (m**-3)

    factor = (
        1.89e-28
        * gaunt
        * Ne_cm ** 2
        / (sqrt(Te) * wavelength_ang ** 2)
        * exp(-12400 / (wavelength_ang * Te))
    ) * reconvert_units

    if zeff is None:
        result = bremsstrahlung / factor
    else:
        result = zeff * factor

    return result




def bremsstrahlung_model(
    Te: ndarray,
    Ne: ndarray,
    zeff: ndarray,
    wavelength: float,
    gaunt_approx="callahan",
) -> ndarray:

    gaunt_funct = {
        "callahan": 1.35 * Te ** 0.15,
        "carson": 7.94425 + 0.5513 * log(Te * wavelength),
    }
    gaunt = gaunt_funct[gaunt_approx]

    constant = 1.89e-53
    te_term =  exp(-1.24e-6 / (wavelength * Te)) / (sqrt(Te) * wavelength ** 2)
    result = constant * zeff * te_term * gaunt * Ne**2
    return result


def bremsstrahlung_jacobian(
    Te: ndarray,
    Ne: ndarray,
    zeff: ndarray,
    wavelength: float,
    gaunt_approx="callahan",
) -> tuple[ndarray, dict[str, ndarray]]:

    # get the gaunt factor and its derivative
    gaunt_funct = {
        "callahan": 1.35 * Te ** 0.15,
        "carson": 7.94425 + 0.5513 * log(Te * wavelength)
    }

    gaunt_grad = {
        "callahan": 0.2025 * Te ** -0.85,
        "carson": 0.5513 / Te
    }

    G = gaunt_funct[gaunt_approx]
    dG_dT = gaunt_grad[gaunt_approx]

    # calculate the temperature-dependant term and its derivative
    constant = 1.89e-53
    k = 1.24e-6 / (wavelength * Te)
    S =  exp(-k) / (sqrt(Te) * wavelength ** 2)
    dS_dT = (k - 0.5) * S / Te

    # calculate the bremsstrahlung and its jacobian
    intermed = constant * S * G * Ne
    dE_dz = intermed * Ne
    dE_dT = constant * zeff * Ne ** 2 * (S * dG_dT + G * dS_dT)
    dE_dN = 2 * zeff * intermed
    bremstrahl = dE_dz * zeff

    jacobian = {
        "te": diagflat(dE_dT),
        "ne": diagflat(dE_dN),
        "z_eff": diagflat(dE_dz),
    }

    return bremstrahl, jacobian


if __name__ == "__main__":
    v1 = zeff_bremsstrahlung(Te=28.5, Ne=2.2e20, wavelength=567, zeff=2.12)
    v2 = bremsstrahlung_model(Te=28.5, Ne=2.2e20, wavelength=567e-9, zeff=2.12)
    print(v1, v2, v1 / v2)

    v1 = zeff_bremsstrahlung(Te=28.5, Ne=2.2e20, wavelength=567, zeff=2.12, gaunt_approx="carson")
    v2 = bremsstrahlung_model(Te=28.5, Ne=2.2e20, wavelength=567e-9, zeff=2.12, gaunt_approx="carson")
    print(v1, v2, v1 / v2)

    from numpy import linspace

    Te = linspace(30, 200, 4)
    Ne = linspace(1e19, 2e20, 4)
    zeff = linspace(1.5, 3.0, 4)
    wl = 567e-9

    eps = 1e-6
    dT = Te * eps
    dN = Ne * eps
    dz = zeff * eps

    dE_dT = 0.5 * (bremsstrahlung_model(Te + dT, Ne, zeff, wl) -  bremsstrahlung_model(Te - dT, Ne, zeff, wl)) / dT
    dE_dN = 0.5 * (bremsstrahlung_model(Te, Ne + dN, zeff, wl) -  bremsstrahlung_model(Te, Ne - dN, zeff, wl)) / dN
    dE_dz = 0.5 * (bremsstrahlung_model(Te, Ne, zeff + dz, wl) -  bremsstrahlung_model(Te, Ne, zeff - dz, wl)) / dz

    _, jac = bremsstrahlung_jacobian(Te, Ne, zeff, wl)

    from numpy import allclose, diagonal
    assert allclose(dE_dT, diagonal(jac["te"]))
    assert allclose(dE_dN, diagonal(jac["ne"]))
    assert allclose(dE_dz, diagonal(jac["z_eff"]))