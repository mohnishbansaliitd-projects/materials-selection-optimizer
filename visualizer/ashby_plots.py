"""
Ashby Property Charts and Pareto Visualization Generator.

Produces publication-grade log-log property space diagrams with material class envelopes,
performance index guide slopes, and Pareto trade-off curves.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List


def setup_plot_style():
    """Configures clean scientific publication style."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 13,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })


CATEGORY_COLORS = {
    "Carbon Steel": "#1f77b4",
    "Alloy Steel": "#aec7e8",
    "Tool Steel": "#17becf",
    "Stainless Steel": "#9467bd",
    "Aluminum Alloy": "#ff7f0e",
    "Titanium Alloy": "#2ca02c",
    "Copper Alloy": "#8c564b",
    "Cast Iron": "#7f7f7f",
    "Magnesium Alloy": "#e377c2",
    "Thermoplastic Polymer": "#bcbd22",
    "High-Perf Polymer": "#dbdb8d",
    "Polymer Composite": "#d62728",
    "Technical Ceramic": "#8c564b"
}


def plot_modulus_vs_density(
    df: pd.DataFrame,
    output_path: str = "outputs/figures/ashby_modulus_density.png"
):
    """
    Generates classical Ashby Chart: Young's Modulus (E) vs Density (rho).
    Includes guide slopes: E/rho (Tension, slope=1), E^1/2/rho (Bending, slope=2), E^1/3/rho (Panel, slope=3).
    """
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)
    
    # Convert density to Mg/m^3 (g/cm^3) for standard Ashby chart representation
    rho = df["density_kg_m3_typ"] / 1000.0
    E = df["youngs_modulus_gpa_typ"]
    
    for cat, group in df.groupby("category"):
        color = CATEGORY_COLORS.get(cat, "#333333")
        ax.scatter(
            group["density_kg_m3_typ"] / 1000.0,
            group["youngs_modulus_gpa_typ"],
            label=cat,
            color=color,
            s=65,
            edgecolors="k",
            alpha=0.85,
            zorder=4
        )
        
    # Annotate select key materials
    key_materials = ["STEEL_4140_QT", "AL_6061_T6", "TI_6AL_4V", "COMPOSITE_CFRP_UD", "CERAMIC_SIC", "POLYMER_PA66"]
    for mat_id in key_materials:
        row = df[df["material_id"] == mat_id]
        if not row.empty:
            r = row.iloc[0]
            ax.annotate(
                r["name"].split()[0],
                (r["density_kg_m3_typ"] / 1000.0, r["youngs_modulus_gpa_typ"]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=7.5,
                fontweight="bold"
            )
            
    # Add guide lines
    rho_range = np.linspace(1.0, 9.0, 100)
    # Slope 1: E / rho = constant -> E = C * rho
    ax.plot(rho_range, 15.0 * rho_range, 'k--', alpha=0.5, label=r"Tie: $M = E/\rho$ (slope 1)")
    # Slope 2: E^1/2 / rho = constant -> E = C * rho^2
    ax.plot(rho_range, 2.0 * (rho_range ** 2), 'b-.', alpha=0.5, label=r"Beam: $M = E^{1/2}/\rho$ (slope 2)")
    # Slope 3: E^1/3 / rho = constant -> E = C * rho^3
    ax.plot(rho_range, 0.4 * (rho_range ** 3), 'm:', alpha=0.5, label=r"Panel: $M = E^{1/3}/\rho$ (slope 3)")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Density $\rho$ [$\mathrm{Mg/m^3}$ or $\mathrm{g/cm^3}$]")
    ax.set_ylabel("Young's Modulus $E$ [GPa]")
    ax.set_title("Ashby Selection Chart: Elastic Modulus vs Density", fontweight="bold")
    ax.set_xlim(0.8, 10.0)
    ax.set_ylim(0.2, 600.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_strength_vs_density(
    df: pd.DataFrame,
    output_path: str = "outputs/figures/ashby_strength_density.png"
):
    """
    Generates Ashby Chart: Yield Strength (sigma_y) vs Density (rho).
    Includes guide slopes: sigma/rho, sigma^(2/3)/rho, sigma^(1/2)/rho.
    """
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)
    
    for cat, group in df.groupby("category"):
        color = CATEGORY_COLORS.get(cat, "#333333")
        ax.scatter(
            group["density_kg_m3_typ"] / 1000.0,
            group["yield_strength_mpa_typ"],
            label=cat,
            color=color,
            s=65,
            edgecolors="k",
            alpha=0.85,
            zorder=4
        )
        
    rho_range = np.linspace(1.0, 9.0, 100)
    ax.plot(rho_range, 60.0 * rho_range, 'k--', alpha=0.5, label=r"Tie: $\sigma/\rho$ (slope 1)")
    ax.plot(rho_range, 15.0 * (rho_range ** 1.5), 'b-.', alpha=0.5, label=r"Beam: $\sigma^{2/3}/\rho$ (slope 1.5)")
    ax.plot(rho_range, 10.0 * (rho_range ** 2.0), 'm:', alpha=0.5, label=r"Panel: $\sigma^{1/2}/\rho$ (slope 2.0)")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Density $\rho$ [$\mathrm{Mg/m^3}$]")
    ax.set_ylabel("Yield Strength $\sigma_y$ [MPa]")
    ax.set_title("Ashby Selection Chart: Strength vs Density", fontweight="bold")
    ax.set_xlim(0.8, 10.0)
    ax.set_ylim(8.0, 4000.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_pareto_front(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    output_path: str
):
    """
    Plots a 2D Pareto trade-off curve with non-dominated front highlighted.
    """
    setup_plot_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Non-Pareto points
    non_pareto = df[~df["is_pareto_optimal"]]
    ax.scatter(
        non_pareto[x_col], non_pareto[y_col],
        c="#999999", s=50, alpha=0.6, label="Dominated Candidates", edgecolors="none"
    )
    
    # Pareto optimal points
    pareto = df[df["is_pareto_optimal"]].sort_values(by=x_col)
    ax.scatter(
        pareto[x_col], pareto[y_col],
        c="#d62728", s=100, alpha=0.9, label="Pareto Optimal Frontier (Rank 1)", edgecolors="k", zorder=5
    )
    ax.plot(pareto[x_col], pareto[y_col], 'r--', alpha=0.7, zorder=4)
    
    for _, row in pareto.iterrows():
        ax.annotate(
            row["name"],
            (row[x_col], row[y_col]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold"
        )
        
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best")
    
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
