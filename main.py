"""
Runs the full materials selection pipeline: data load, imputation benchmark,
Ashby charts, gearbox case study, and Pareto trade-off plot.
"""

import os
import sys
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from core.ashby_indices import calculate_index, list_available_indices
from core.screening import MaterialScreener
from core.pareto_optimizer import find_pareto_front, calculate_topsis_score
from core.uncertainty import run_monte_carlo_rank_stability
from core.imputation import physics_informed_imputation, evaluate_grouped_imputation
from core.geometry_optimizer import optimize_shaft_diameter, DEFAULT_TARGET_TORQUE_NM
from case_study.gearbox_analysis import analyze_gear_material, analyze_shaft_material, analyze_housing_material
from visualizer.ashby_plots import plot_modulus_vs_density, plot_strength_vs_density, plot_pareto_front, plot_shaft_geometry_pareto


def run_full_pipeline():
    print("Running materials selection optimizer")

    data_path = os.path.join(PROJECT_DIR, "data", "materials_database.csv")
    df = pd.read_csv(data_path)
    print(f"\n[1] Loaded materials database: {len(df)} materials across {df['category'].nunique()} categories.")

    print("\n[2] Evaluating imputation engine (physics vs grouped ML)...")
    df_imputed = physics_informed_imputation(df)
    imp_metrics = evaluate_grouped_imputation(df_imputed)
    print(f"    - Grouped leave-one-alloy-out Random Forest MAE: {imp_metrics['rf_grouped_mae']:.2f} MPa (R²: {imp_metrics['rf_grouped_r2']:.2f})")
    print(f"    - Grouped leave-one-alloy-out k-NN MAE: {imp_metrics['knn_grouped_mae']:.2f} MPa (R²: {imp_metrics['knn_grouped_r2']:.2f})")

    print("\n[3] Generating Ashby charts...")
    figures_dir = os.path.join(PROJECT_DIR, "outputs", "figures")
    os.makedirs(figures_dir, exist_ok=True)

    plot_modulus_vs_density(df, output_path=os.path.join(figures_dir, "ashby_modulus_density.png"))
    plot_strength_vs_density(df, output_path=os.path.join(figures_dir, "ashby_strength_density.png"))
    print(f"    - Saved charts to {figures_dir}")

    print("\n[4] Running gearbox case study...")

    res_gear = analyze_gear_material(df)
    print(f"\n--- Component 1: {res_gear['component']} ---")
    print(f"Rationale: {res_gear['rationale']}")
    print("\nTop 3 recommended materials (TOPSIS ranking):")
    print(res_gear['top_recommendations'][['name', 'category', 'hardness_hb_typ', 'cost_usd_per_kg_typ', 'topsis_score']].head(3).to_string(index=False))

    res_shaft = analyze_shaft_material(df)
    print(f"\n--- Component 2: {res_shaft['component']} ---")
    print(f"Rationale: {res_shaft['rationale']}")
    print("\nTop 3 recommended materials (TOPSIS ranking):")
    print(res_shaft['top_recommendations'][['name', 'category', 'yield_strength_mpa_typ', 'cost_usd_per_kg_typ', 'topsis_score']].head(3).to_string(index=False))

    res_housing = analyze_housing_material(df)
    print(f"\n--- Component 3: {res_housing['component']} ---")
    print(f"Rationale: {res_housing['rationale']}")
    print("\nTop 3 recommended materials (TOPSIS ranking):")
    print(res_housing['top_recommendations'][['name', 'category', 'density_kg_m3_typ', 'cost_usd_per_kg_typ', 'topsis_score']].head(3).to_string(index=False))

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
    print(f"\n[5] Saved Pareto trade-off plot to {os.path.join(figures_dir, 'pareto_gear_cost_strength.png')}")

    print("\n[6] Shaft Geometry Optimization (pymoo NSGA-II)...")
    top_shaft = res_shaft["top_recommendations"].iloc[0]
    top_shaft_props = df[df["material_id"] == top_shaft["material_id"]].iloc[0]
    print(f"    - Optimizing solid circular shaft diameter for top-ranked candidate: {top_shaft['name']}")
    print(f"    - Assumed design load: {DEFAULT_TARGET_TORQUE_NM:.0f} N·m transmitted torque (no torque/power spec exists in this codebase; see core/geometry_optimizer.py docstring)")

    shaft_pareto = optimize_shaft_diameter(
        sigma_y_mpa=top_shaft_props["yield_strength_mpa_typ"],
        density_kg_m3=top_shaft_props["density_kg_m3_typ"],
        pop_size=40,
        n_gen=60
    )
    print("\nPareto-optimal diameter / mass / safety factor trade-off:")
    print(shaft_pareto.to_string(index=False))

    plot_shaft_geometry_pareto(
        shaft_pareto,
        material_name=top_shaft["name"],
        target_torque_nm=DEFAULT_TARGET_TORQUE_NM,
        output_path=os.path.join(figures_dir, "shaft_geometry_pareto.png")
    )
    print(f"\n    - Saved shaft geometry Pareto plot to {os.path.join(figures_dir, 'shaft_geometry_pareto.png')}")

    print("\nDone.")


if __name__ == "__main__":
    run_full_pipeline()
