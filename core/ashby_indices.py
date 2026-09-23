"""
Ashby performance indices, following Ashby's "Materials Selection in Mechanical
Design" derivations for minimum-mass design under stiffness/strength constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union


ASHBY_INDICES = {
    "tie_stiff_mass": {
        "name": "Lightweight Stiff Tie (Tension)",
        "formula": "E / rho",
        "power_e": 1.0,
        "power_sig": 0.0,
        "power_rho": -1.0,
        "mode": "max",
        "description": "Minimize mass for a tensile member with target axial stiffness."
    },
    "tie_strong_mass": {
        "name": "Lightweight Strong Tie (Tension)",
        "formula": "sigma_y / rho",
        "power_e": 0.0,
        "power_sig": 1.0,
        "power_rho": -1.0,
        "mode": "max",
        "description": "Minimize mass for a tensile member with target load capacity."
    },
    "beam_stiff_mass": {
        "name": "Lightweight Stiff Beam (Bending)",
        "formula": "E^(1/2) / rho",
        "power_e": 0.5,
        "power_sig": 0.0,
        "power_rho": -1.0,
        "mode": "max",
        "description": "Minimize mass for a beam in bending with target deflection/stiffness."
    },
    "beam_strong_mass": {
        "name": "Lightweight Strong Beam (Bending)",
        "formula": "sigma_y^(2/3) / rho",
        "power_e": 0.0,
        "power_sig": 2.0 / 3.0,
        "power_rho": -1.0,
        "mode": "max",
        "description": "Minimize mass for a beam in bending with target failure load (yield)."
    },
    "shaft_stiff_mass": {
        "name": "Lightweight Stiff Shaft (Torsion)",
        "formula": "G^(1/2) / rho",
        "power_e": 0.0,
        "power_sig": 0.0,
        "power_rho": -1.0,
        "mode": "max",
        "description": "Minimize mass for a solid circular shaft with target torsional rigidity."
    },
    "shaft_strong_mass": {
        "name": "Lightweight Strong Shaft (Torsion)",
        "formula": "sigma_y^(2/3) / rho",
        "power_e": 0.0,
        "power_sig": 2.0 / 3.0,
        "power_rho": -1.0,
        "mode": "max",
        "description": "Minimize mass for a solid circular shaft with target torque capacity."
    },
    "panel_stiff_mass": {
        "name": "Lightweight Stiff Plate / Panel (Bending)",
        "formula": "E^(1/3) / rho",
        "power_e": 1.0 / 3.0,
        "power_sig": 0.0,
        "power_rho": -1.0,
        "mode": "max",
        "description": "Minimize mass for a flat panel/plate in bending (e.g. gearbox casing wall)."
    },
    "panel_strong_mass": {
        "name": "Lightweight Strong Plate / Panel (Bending)",
        "formula": "sigma_y^(1/2) / rho",
        "power_e": 0.0,
        "power_sig": 0.5,
        "power_rho": -1.0,
        "mode": "max",
        "description": "Minimize mass for a flat panel with target yield/strength under transverse load."
    },
    "column_buckling_mass": {
        "name": "Lightweight Column (Euler Buckling)",
        "formula": "E^(1/2) / rho",
        "power_e": 0.5,
        "power_sig": 0.0,
        "power_rho": -1.0,
        "mode": "max",
        "description": "Minimize mass for slender compressive column subject to elastic buckling."
    },
    "spring_energy_mass": {
        "name": "Maximum Elastic Energy Storage (Springs)",
        "formula": "sigma_y^2 / (E * rho)",
        "power_e": -1.0,
        "power_sig": 2.0,
        "power_rho": -1.0,
        "mode": "max",
        "description": "Maximize resilient strain energy per unit mass before yielding."
    },
    "beam_strong_cost": {
        "name": "Cost-Effective Strong Beam (Bending)",
        "formula": "sigma_y^(2/3) / (cost * rho)",
        "power_e": 0.0,
        "power_sig": 2.0 / 3.0,
        "power_rho": -1.0,
        "mode": "max",
        "description": "Minimize total raw material cost for a beam with target bending strength."
    },
    "gear_contact_fatigue": {
        "name": "Gear Tooth Pitting & Wear Resistance",
        "formula": "Hardness_HB / (cost * rho)^(1/3)",
        "mode": "max",
        "description": "Maximize contact fatigue resistance and wear life under cyclic tooth meshing."
    }
}


def calculate_index(
    df: pd.DataFrame,
    index_key: str,
    density_col: str = "density_kg_m3_typ",
    modulus_col: str = "youngs_modulus_gpa_typ",
    yield_col: str = "yield_strength_mpa_typ",
    cost_col: str = "cost_usd_per_kg_typ",
    hardness_col: str = "hardness_hb_typ",
    shear_col: str = "shear_modulus_gpa_typ"
) -> pd.Series:
    """Density is converted GPa/(g/cm^3) style so indices stay dimensionally comparable."""
    if index_key not in ASHBY_INDICES:
        raise ValueError(f"Unknown index_key: '{index_key}'. Available: {list(ASHBY_INDICES.keys())}")

    rho = df[density_col].astype(float)
    E = df[modulus_col].astype(float)
    sig_y = df[yield_col].astype(float)
    cost = df[cost_col].astype(float) if cost_col in df else pd.Series(1.0, index=df.index)
    HB = df[hardness_col].astype(float) if hardness_col in df else pd.Series(100.0, index=df.index)
    G = df[shear_col].astype(float) if shear_col in df else E / (2.0 * (1.0 + 0.3))

    if index_key == "tie_stiff_mass":
        return E * 1e3 / (rho / 1e3)  # proportional (GPa / (g/cm3))
    elif index_key == "tie_strong_mass":
        return sig_y / (rho / 1e3)
    elif index_key == "beam_stiff_mass":
        return (E ** 0.5) / (rho / 1e3)
    elif index_key == "beam_strong_mass":
        return (sig_y ** (2.0 / 3.0)) / (rho / 1e3)
    elif index_key == "shaft_stiff_mass":
        return (G ** 0.5) / (rho / 1e3)
    elif index_key == "shaft_strong_mass":
        return ((0.577 * sig_y) ** (2.0 / 3.0)) / (rho / 1e3)
    elif index_key == "panel_stiff_mass":
        return (E ** (1.0 / 3.0)) / (rho / 1e3)
    elif index_key == "panel_strong_mass":
        return (sig_y ** 0.5) / (rho / 1e3)
    elif index_key == "column_buckling_mass":
        return (E ** 0.5) / (rho / 1e3)
    elif index_key == "spring_energy_mass":
        return (sig_y ** 2.0) / (E * (rho / 1e3))
    elif index_key == "beam_strong_cost":
        return (sig_y ** (2.0 / 3.0)) / (cost * (rho / 1e3))
    elif index_key == "gear_contact_fatigue":
        return HB / ((cost * (rho / 1e3)) ** (1.0 / 3.0))
    else:
        raise NotImplementedError(f"Index formula not mapped: {index_key}")


def list_available_indices() -> Dict[str, str]:
    """Returns dictionary of index keys and descriptions."""
    return {k: f"{v['name']} ({v['formula']}) - {v['description']}" for k, v in ASHBY_INDICES.items()}
