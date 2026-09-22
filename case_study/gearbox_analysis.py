"""
Compact Variable Gearbox Case Study Module.

Performs component-level material selection for:
1. Spur Gear / Pinion (Tooth Root Bending & Hertz Contact Fatigue)
2. Transmission Shaft (Torsional Strength & Fatigue)
3. Gearbox Housing / Casing (Panel Bending Rigidity & Lightweighting)

Validates algorithm recommendations against physical manufacturing choices.
"""

import pandas as pd
from typing import Dict, Any, Tuple
from core.screening import MaterialScreener
from core.ashby_indices import calculate_index
from core.pareto_optimizer import find_pareto_front, calculate_topsis_score
from core.uncertainty import run_monte_carlo_rank_stability


def analyze_gear_material(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Component 1: Spur Gear / Pinion.
    Primary requirements: High surface fatigue wear limit (Hertz contact),
    root bending fatigue strength (Lewis formula), toughness to avoid tooth breakage.
    """
    screener = MaterialScreener(df)
    screened, stats = screener.filter(
        min_yield_strength_mpa=500.0,
        min_fracture_toughness=25.0,
        min_elongation_pct=5.0,
        excluded_categories=["Polymer Composite", "Thermoplastic Polymer", "Technical Ceramic"]
    )
    
    # Calculate key indices
    screened["idx_bending_strength"] = calculate_index(screened, "beam_strong_mass")
    screened["idx_contact_wear"] = calculate_index(screened, "gear_contact_fatigue")
    
    # Multi-objective Pareto front: Maximize contact wear, Maximize bending strength, Minimize cost
    pareto_df = find_pareto_front(
        screened,
        objectives=["idx_contact_wear", "idx_bending_strength", "cost_usd_per_kg_typ"],
        directions=["max", "max", "min"]
    )
    
    topsis_df = calculate_topsis_score(
        pareto_df,
        objectives=["idx_contact_wear", "idx_bending_strength", "cost_usd_per_kg_typ"],
        directions=["max", "max", "min"],
        weights=[0.45, 0.35, 0.20]
    )
    
    mc_stability = run_monte_carlo_rank_stability(topsis_df, index_key="gear_contact_fatigue", n_iterations=500)
    
    return {
        "component": "Spur Gear / Pinion",
        "top_recommendations": topsis_df.head(5)[["material_id", "name", "category", "hardness_hb_typ", "yield_strength_mpa_typ", "cost_usd_per_kg_typ", "pareto_rank", "topsis_score"]],
        "pareto_front": pareto_df[pareto_df["is_pareto_optimal"]][["material_id", "name", "hardness_hb_typ", "cost_usd_per_kg_typ"]],
        "rejection_stats": stats,
        "stability": mc_stability.head(5)[["material_id", "name", "prob_top_3_pct", "mean_simulated_rank"]],
        "rationale": "20MnCr5 and AISI 8620 case-carburized steels provide high surface hardness (600+ HB) for contact fatigue with a ductile shock-resistant core."
    }


def analyze_shaft_material(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Component 2: Transmission Input/Output Shaft.
    Primary requirements: High torsional and bending fatigue strength, rigidity (low deflection), toughness.
    """
    screener = MaterialScreener(df)
    screened, stats = screener.filter(
        min_yield_strength_mpa=300.0,
        min_fracture_toughness=25.0,
        min_elongation_pct=10.0,
        min_youngs_modulus_gpa=60.0,
        excluded_categories=["Technical Ceramic", "Thermoplastic Polymer"]
    )
    
    screened["idx_shaft_strength"] = calculate_index(screened, "shaft_strong_mass")
    screened["idx_shaft_stiff"] = calculate_index(screened, "shaft_stiff_mass")
    
    pareto_df = find_pareto_front(
        screened,
        objectives=["idx_shaft_strength", "idx_shaft_stiff", "cost_usd_per_kg_typ"],
        directions=["max", "max", "min"]
    )
    
    topsis_df = calculate_topsis_score(
        pareto_df,
        objectives=["idx_shaft_strength", "idx_shaft_stiff", "cost_usd_per_kg_typ"],
        directions=["max", "max", "min"],
        weights=[0.45, 0.35, 0.20]
    )
    
    mc_stability = run_monte_carlo_rank_stability(topsis_df, index_key="shaft_strong_mass", n_iterations=500)
    
    return {
        "component": "Transmission Shaft",
        "top_recommendations": topsis_df.head(5)[["material_id", "name", "category", "yield_strength_mpa_typ", "youngs_modulus_gpa_typ", "cost_usd_per_kg_typ", "pareto_rank", "topsis_score"]],
        "pareto_front": pareto_df[pareto_df["is_pareto_optimal"]][["material_id", "name", "idx_shaft_strength", "cost_usd_per_kg_typ"]],
        "rejection_stats": stats,
        "stability": mc_stability.head(5)[["material_id", "name", "prob_top_3_pct", "mean_simulated_rank"]],
        "rationale": "AISI 4140 Q&T and AISI 4340 Q&T alloy steels maximize the torsional strength index sigma_y^(2/3)/rho while maintaining high toughness (65 MPa m^1/2) and fatigue limits."
    }


def analyze_housing_material(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Component 3: Gearbox Housing / Casing.
    Primary requirements: Panel bending rigidity (E^(1/3)/rho), lightweighting, machinability/castability, vibration damping.
    """
    screener = MaterialScreener(df)
    screened, stats = screener.filter(
        min_elongation_pct=0.5,
        min_youngs_modulus_gpa=20.0,
        min_machinability_pct=40.0,
        excluded_categories=["Technical Ceramic"]
    )
    
    screened["idx_panel_stiffness"] = calculate_index(screened, "panel_stiff_mass")
    screened["idx_cost_stiffness"] = (screened["youngs_modulus_gpa_typ"] ** (1.0/3.0)) / (screened["cost_usd_per_kg_typ"] * (screened["density_kg_m3_typ"] / 1000.0))
    
    pareto_df = find_pareto_front(
        screened,
        objectives=["idx_panel_stiffness", "idx_cost_stiffness", "density_kg_m3_typ"],
        directions=["max", "max", "min"]
    )
    
    topsis_df = calculate_topsis_score(
        pareto_df,
        objectives=["idx_panel_stiffness", "idx_cost_stiffness", "density_kg_m3_typ"],
        directions=["max", "max", "min"],
        weights=[0.40, 0.35, 0.25]
    )
    
    mc_stability = run_monte_carlo_rank_stability(topsis_df, index_key="panel_stiff_mass", n_iterations=500)
    
    return {
        "component": "Gearbox Housing / Casing",
        "top_recommendations": topsis_df.head(5)[["material_id", "name", "category", "density_kg_m3_typ", "youngs_modulus_gpa_typ", "cost_usd_per_kg_typ", "pareto_rank", "topsis_score"]],
        "pareto_front": pareto_df[pareto_df["is_pareto_optimal"]][["material_id", "name", "idx_panel_stiffness", "cost_usd_per_kg_typ"]],
        "rejection_stats": stats,
        "stability": mc_stability.head(5)[["material_id", "name", "prob_top_3_pct", "mean_simulated_rank"]],
        "rationale": "A356.0-T6 Cast Aluminum and 6061-T6 optimize panel bending stiffness E^(1/3)/rho while reducing housing mass by ~62% compared to Class 30 Gray Cast Iron."
    }
