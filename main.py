"""
Main Runner for Project 1: Data-Driven Materials Selection Optimizer.

Executes:
1. Materials database ingestion & verification.
2. Physics-informed & ML grouped imputation benchmark.
3. Classic Ashby property chart generation.
4. Compact Variable Gearbox Case Study (Gears, Shaft, Housing).
5. Monte Carlo rank stability and uncertainty analysis.
"""

import os
import sys
import pandas as pd

# Add project root to sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from core.ashby_indices import calculate_index, list_available_indices
from core.screening import MaterialScreener
from core.pareto_optimizer import find_pareto_front, calculate_topsis_score
from core.uncertainty import run_monte_carlo_rank_stability
from core.imputation import physics_informed_imputation, evaluate_grouped_imputation
from case_study.gearbox_analysis import analyze_gear_material, analyze_shaft_material, analyze_housing_material
from visualizer.ashby_plots import plot_modulus_vs_density, plot_strength_vs_density, plot_pareto_front


def run_full_pipeline():
    print("=" * 80)
    print("🚀 RUNNING PROJECT 1: DATA-DRIVEN MATERIALS SELECTION OPTIMIZER")
    print("=" * 80)
    
    # 1. Load Data
    data_path = os.path.join(PROJECT_DIR, "data", "materials_database.csv")
    df = pd.read_csv(data_path)
    print(f"\n[1] Loaded Materials Database: {len(df)} materials across {df['category'].nunique()} categories.")
    
    # 2. Physics & ML Imputation Test
    print("\n[2] Evaluating Imputation Engine (Physics vs Grouped ML)...")
    df_imputed = physics_informed_imputation(df)
    imp_metrics = evaluate_grouped_imputation(df_imputed)
    print(f"    - Grouped Leave-One-Alloy-Out Random Forest MAE: {imp_metrics['rf_grouped_mae']:.2f} MPa (R²: {imp_metrics['rf_grouped_r2']:.2f})")
    print(f"    - Grouped Leave-One-Alloy-Out k-NN MAE: {imp_metrics['knn_grouped_mae']:.2f} MPa (R²: {imp_metrics['knn_grouped_r2']:.2f})")
    
    # 3. Generate Ashby Charts
    print("\n[3] Generating Publication-Grade Ashby Charts...")
    figures_dir = os.path.join(PROJECT_DIR, "outputs", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    plot_modulus_vs_density(df, output_path=os.path.join(figures_dir, "ashby_modulus_density.png"))
    plot_strength_vs_density(df, output_path=os.path.join(figures_dir, "ashby_strength_density.png"))
    print(f"    - Saved charts to {figures_dir}")
    
    # 4. Gearbox Case Study Execution
    print("\n[4] Executing Compact Variable Gearbox Case Study...")
    
    # Component A: Gears
    res_gear = analyze_gear_material(df)
    print(f"\n--- Component 1: {res_gear['component']} ---")
    print(f"Rationale: {res_gear['rationale']}")
    print("\nTop 3 Recommended Materials (TOPSIS Multi-Criteria Ranking):")
    print(res_gear['top_recommendations'][['name', 'category', 'hardness_hb_typ', 'cost_usd_per_kg_typ', 'topsis_score']].head(3).to_string(index=False))
    
    # Component B: Shaft
    res_shaft = analyze_shaft_material(df)
    print(f"\n--- Component 2: {res_shaft['component']} ---")
    print(f"Rationale: {res_shaft['rationale']}")
    print("\nTop 3 Recommended Materials (TOPSIS Multi-Criteria Ranking):")
    print(res_shaft['top_recommendations'][['name', 'category', 'yield_strength_mpa_typ', 'cost_usd_per_kg_typ', 'topsis_score']].head(3).to_string(index=False))
    
    # Component C: Housing
    res_housing = analyze_housing_material(df)
    print(f"\n--- Component 3: {res_housing['component']} ---")
    print(f"Rationale: {res_housing['rationale']}")
    print("\nTop 3 Recommended Materials (TOPSIS Multi-Criteria Ranking):")
    print(res_housing['top_recommendations'][['name', 'category', 'density_kg_m3_typ', 'cost_usd_per_kg_typ', 'topsis_score']].head(3).to_string(index=False))
    
    # 5. Pareto Frontier Plot
    screened_gear, _ = MaterialScreener(df).filter(
        min_yield_strength_mpa=500.0,
        min_fracture_toughness=25.0,
        excluded_categories=["Polymer Composite", "Thermoplastic Polymer", "Technical Ceramic"]
    )
    screened_gear["idx_bending_strength"] = calculate_index(screened_gear, "beam_strong_mass")
    pareto_gear = find_pareto_front(
        screened_gear,
        objectives=["idx_bending_strength", "cost_usd_per_kg_typ"],
        directions=["max", "min"]
    )
    plot_pareto_front(
        pareto_gear,
        x_col="cost_usd_per_kg_typ",
        y_col="idx_bending_strength",
        x_label="Raw Stock Cost [$ / kg]",
        y_label=r"Bending Strength Index $\sigma_y^{2/3}/\rho$",
        title="Pareto Trade-off: Gear Strength Index vs Cost",
        output_path=os.path.join(figures_dir, "pareto_gear_cost_strength.png")
    )
    print(f"\n[5] Saved Pareto Trade-off plot to {os.path.join(figures_dir, 'pareto_gear_cost_strength.png')}")
    
    print("\n" + "=" * 80)
    print("✅ PROJECT 1 EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_pipeline()
