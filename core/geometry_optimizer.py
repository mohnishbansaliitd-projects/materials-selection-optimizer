"""
Shaft diameter optimization via pymoo's NSGA-II -- a continuous design variable for one
already-selected material, trading off mass against torsional safety factor. Complements
core/pareto_optimizer.py, which ranks the fixed discrete material table instead.

No torque/power spec exists elsewhere in this codebase, so target torque (400 N*m, a mid-size
gearbox output shaft) and shaft length (300 mm bearing span) are assumed, not measured -- both
are kwargs so a real spec can override them.
"""

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

DEFAULT_TARGET_TORQUE_NM = 400.0
DEFAULT_SHAFT_LENGTH_MM = 300.0
DEFAULT_D_MIN_MM = 10.0
DEFAULT_D_MAX_MM = 60.0


class ShaftGeometryProblem(ElementwiseProblem):
    """Solid circular shaft: minimize mass, maximize torsional safety factor.

    mass = (pi/4) * d^2 * L * rho  (uniform solid cylinder)
    torque_capacity = (pi/16) * d^3 * tau_y, tau_y = 0.577 * sigma_y (von Mises
    shear yield), matching the (0.577*sigma_y)^(2/3) term already used in
    core/ashby_indices.py's 'shaft_strong_mass' index.
    safety_factor = torque_capacity / target_torque
    """

    def __init__(
        self,
        sigma_y_mpa: float,
        density_kg_m3: float,
        target_torque_nm: float = DEFAULT_TARGET_TORQUE_NM,
        shaft_length_mm: float = DEFAULT_SHAFT_LENGTH_MM,
        d_min_mm: float = DEFAULT_D_MIN_MM,
        d_max_mm: float = DEFAULT_D_MAX_MM,
    ):
        super().__init__(n_var=1, n_obj=2, n_constr=0, xl=np.array([d_min_mm]), xu=np.array([d_max_mm]))
        self.sigma_y_mpa = sigma_y_mpa
        self.density_kg_m3 = density_kg_m3
        self.target_torque_nm = target_torque_nm
        self.shaft_length_mm = shaft_length_mm

    def _evaluate(self, x, out, *args, **kwargs):
        d_m = x[0] / 1000.0
        L_m = self.shaft_length_mm / 1000.0

        mass_kg = (np.pi / 4.0) * (d_m ** 2) * L_m * self.density_kg_m3

        tau_y_pa = 0.577 * self.sigma_y_mpa * 1e6
        torque_capacity_nm = (np.pi / 16.0) * (d_m ** 3) * tau_y_pa
        safety_factor = torque_capacity_nm / self.target_torque_nm

        out["F"] = [mass_kg, -safety_factor]


def optimize_shaft_diameter(
    sigma_y_mpa: float,
    density_kg_m3: float,
    target_torque_nm: float = DEFAULT_TARGET_TORQUE_NM,
    shaft_length_mm: float = DEFAULT_SHAFT_LENGTH_MM,
    d_min_mm: float = DEFAULT_D_MIN_MM,
    d_max_mm: float = DEFAULT_D_MAX_MM,
    pop_size: int = 40,
    n_gen: int = 60,
    seed: int = 1,
) -> pd.DataFrame:
    """Runs NSGA-II over shaft diameter and returns the Pareto-optimal set,
    sorted by diameter, with mass, torque capacity and safety factor columns."""
    problem = ShaftGeometryProblem(
        sigma_y_mpa=sigma_y_mpa,
        density_kg_m3=density_kg_m3,
        target_torque_nm=target_torque_nm,
        shaft_length_mm=shaft_length_mm,
        d_min_mm=d_min_mm,
        d_max_mm=d_max_mm,
    )

    algorithm = NSGA2(pop_size=pop_size)
    result = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=False)

    diameters_mm = result.X[:, 0]
    mass_kg = result.F[:, 0]
    safety_factor = -result.F[:, 1]
    torque_capacity_nm = safety_factor * target_torque_nm

    pareto_df = pd.DataFrame({
        "diameter_mm": diameters_mm,
        "mass_kg": mass_kg,
        "torque_capacity_nm": torque_capacity_nm,
        "safety_factor": safety_factor,
    }).sort_values(by="diameter_mm").reset_index(drop=True)

    return pareto_df
