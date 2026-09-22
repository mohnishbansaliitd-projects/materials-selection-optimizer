"""
Unit Test Suite for Project 1: Data-Driven Materials Selection Optimizer.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.ashby_indices import calculate_index, ASHBY_INDICES
from core.screening import MaterialScreener
from core.pareto_optimizer import find_pareto_front, calculate_topsis_score
from core.uncertainty import run_monte_carlo_rank_stability
from core.imputation import physics_informed_imputation


@pytest.fixture
def sample_dataset():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "materials_database.csv")
    return pd.read_csv(data_path)


def test_ashby_indices_calculation(sample_dataset):
    """Verifies that all registered Ashby indices compute finite, positive values."""
    for idx_key in ASHBY_INDICES.keys():
        vals = calculate_index(sample_dataset, idx_key)
        assert len(vals) == len(sample_dataset)
        assert not vals.isna().any(), f"NaN found in index {idx_key}"
        assert (vals > 0).all(), f"Non-positive value in index {idx_key}"


def test_screening_filter(sample_dataset):
    """Tests hard constraint screening logic."""
    screener = MaterialScreener(sample_dataset)
    screened, stats = screener.filter(
        min_yield_strength_mpa=500.0,
        min_fracture_toughness=30.0,
        allowed_categories=["Alloy Steel", "Stainless Steel", "Aluminum Alloy"]
    )
    assert len(screened) > 0
    assert (screened["yield_strength_mpa_typ"] >= 500.0).all()
    assert (screened["fracture_toughness_mpam12_typ"] >= 30.0).all()
    assert screened["category"].isin(["Alloy Steel", "Stainless Steel", "Aluminum Alloy"]).all()


def test_pareto_front_logic():
    """Verifies 2D non-dominated sorting on known synthetic points."""
    df_toy = pd.DataFrame({
        "material_id": ["A", "B", "C", "D"],
        "cost": [10.0, 20.0, 15.0, 30.0],
        "mass": [5.0, 2.0, 6.0, 8.0]
    })
    # Minimize both cost and mass
    res = find_pareto_front(df_toy, objectives=["cost", "mass"], directions=["min", "min"])
    # Point A (10, 5) and Point B (20, 2) are non-dominated.
    # Point C (15, 6) is dominated by A (10 < 15 and 5 < 6).
    # Point D (30, 8) is dominated by all.
    pareto_members = res[res["is_pareto_optimal"]]["material_id"].tolist()
    assert "A" in pareto_members
    assert "B" in pareto_members
    assert "C" not in pareto_members
    assert "D" not in pareto_members


def test_topsis_scoring(sample_dataset):
    """Verifies that TOPSIS scores are bounded between 0 and 1."""
    df_eval = sample_dataset.copy()
    df_eval["idx_strength"] = calculate_index(df_eval, "beam_strong_mass")
    topsis_res = calculate_topsis_score(
        df_eval,
        objectives=["idx_strength", "cost_usd_per_kg_typ"],
        directions=["max", "min"],
        weights=[0.7, 0.3]
    )
    assert "topsis_score" in topsis_res.columns
    assert (topsis_res["topsis_score"] >= 0.0).all()
    assert (topsis_res["topsis_score"] <= 1.0).all()


def test_monte_carlo_stability(sample_dataset):
    """Verifies Monte Carlo simulation generates valid probability distributions."""
    mc_res = run_monte_carlo_rank_stability(sample_dataset, index_key="beam_strong_mass", n_iterations=50)
    assert "prob_top_3_pct" in mc_res.columns
    assert (mc_res["prob_top_3_pct"] >= 0.0).all()
    assert (mc_res["prob_top_3_pct"] <= 100.0).all()


def test_physics_imputation():
    """Verifies isotropic elasticity shear modulus calculation."""
    df_missing = pd.DataFrame({
        "youngs_modulus_gpa_typ": [200.0],
        "poissons_ratio_typ": [0.25],
        "shear_modulus_gpa_typ": [np.nan],
        "tensile_strength_mpa_typ": [700.0],
        "hardness_hb_typ": [np.nan]
    })
    imputed = physics_informed_imputation(df_missing)
    expected_G = 200.0 / (2.0 * (1.0 + 0.25))  # 80.0
    assert np.isclose(imputed.loc[0, "shear_modulus_gpa_typ"], expected_G)
    assert np.isclose(imputed.loc[0, "hardness_hb_typ"], 700.0 / 3.45)


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "materials_database.csv")
    df_test = pd.read_csv(data_path)
    
    print("Running test_ashby_indices_calculation...")
    test_ashby_indices_calculation(df_test)
    print("Running test_screening_filter...")
    test_screening_filter(df_test)
    print("Running test_pareto_front_logic...")
    test_pareto_front_logic()
    print("Running test_topsis_scoring...")
    test_topsis_scoring(df_test)
    print("Running test_monte_carlo_stability...")
    test_monte_carlo_stability(df_test)
    print("Running test_physics_imputation...")
    test_physics_imputation()
    print("\n✅ ALL UNIT TESTS PASSED!")
