from numpy import ndarray, exp, sqrt, log, eye
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

    def predictions(self, fields) -> ndarray:
        return fields[self.field_name]

    def predictions_and_jacobians(
            self, fields
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
        return zeff_bremsstrahlung(
            Te=te,
            Ne=ne,
            wavelength=self.wavelength,
            zeff=z_eff,
        )

    def predictions_and_jacobians(
            self, te: ndarray, ne: ndarray, z_eff: ndarray
    ) -> tuple[ndarray, dict[str, ndarray]]:
        raise NotImplementedError()


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




def zeff_bremsstrahlung_alt(
    Te: ndarray,
    Ne: ndarray,
    zeff: ndarray,
    wavelength: float,
    gaunt_approx="callahan",
) -> ndarray:

    gaunt_funct = {
        "callahan": 1.35 * Te ** 0.15,
        "carson": -4.7499 + 0.5513 * log(Te * wavelength) + 0.5513 * log(1e10),
    }
    gaunt = gaunt_funct[gaunt_approx]

    constant = 1.89e-53
    te_term =  exp(-1.24e-6 / (wavelength * Te)) / (sqrt(Te) * wavelength ** 2)
    result = constant * zeff * te_term * gaunt * Ne**2
    return result


def zeff_bremsstrahlung_jacobian(
    Te: ndarray,
    Ne: ndarray,
    zeff: ndarray,
    wavelength: float,
    gaunt_approx="callahan",
) -> ndarray:

    gaunt_funct = {
        "callahan": 1.35 * Te ** 0.15,
        "carson": -4.7499 + 0.5513 * log(Te * wavelength) + 0.5513 * log(1e10),
    }
    gaunt = gaunt_funct[gaunt_approx]

    constant = 1.89e-53
    te_term =  exp(-1.24e-6 / (wavelength * Te)) / (sqrt(Te) * wavelength ** 2)
    result = constant * zeff * te_term * gaunt * Ne**2

    jacobian = {
        "te": ...,
        "ne": 2 * constant * zeff * te_term * gaunt * Ne,
        "zeff": constant * te_term * gaunt * Ne**2,
    }

    return result


if __name__ == "__main__":
    v1 = zeff_bremsstrahlung(Te=28.5, Ne=2.2e20, wavelength=567, zeff=2.12)
    v2 = zeff_bremsstrahlung_alt(Te=28.5, Ne=2.2e20, wavelength=567e-9, zeff=2.12)
    print(v1, v2, v1 / v2)

    import matplotlib.pyplot as plt
    from numpy import linspace

    wavelength = 567e-9
    Te = linspace(1, 200, 200)
    cal = 1.35 * Te ** 0.15
    car = -4.7499 + 0.5513 * log(Te * wavelength) + 0.5513 * log(1e10)


    plt.plot(Te, cal)
    plt.plot(Te, car)
    plt.grid()
    plt.show()

    plt.plot(Te, cal / car)
    plt.yscale("log")
    plt.grid()
    plt.show()