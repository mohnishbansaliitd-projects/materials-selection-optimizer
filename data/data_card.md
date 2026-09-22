# Materials Selection Dataset Data Card

## 1. Dataset Overview
- **Name**: Curated Engineering Materials Selection Dataset (v1.0)
- **Domain**: Mechanical & Materials Engineering Design (Metals, Polymers, Composites, Ceramics)
- **Primary Use**: Systematic screening, Ashby index ranking, and multi-objective Pareto optimization for mechanical components (e.g. gearboxes, shafts, structural frames).
- **Format**: CSV format (`materials_database.csv`)

## 2. Properties & Column Definitions
- `material_id`: Unique identifier key.
- `name`: Standard engineering alloy/material designation.
- `category`: Broad material family (Carbon Steel, Alloy Steel, Stainless Steel, Aluminum Alloy, Titanium Alloy, Copper Alloy, Cast Iron, Magnesium Alloy, Thermoplastic Polymer, High-Perf Polymer, Polymer Composite, Technical Ceramic).
- `condition`: Heat treatment or processing temper (e.g., Q&T, T6, Cold Drawn, Annealed).
- `density_kg_m3_min`, `_max`, `_typ`: Mass density in $\text{kg/m}^3$.
- `youngs_modulus_gpa_min`, `_max`, `_typ`: Tensile elastic modulus $E$ in $\text{GPa}$.
- `shear_modulus_gpa_typ`: Shear modulus $G$ in $\text{GPa}$.
- `poissons_ratio_typ`: Poisson's ratio $\nu$ (dimensionless).
- `yield_strength_mpa_min`, `_max`, `_typ`: 0.2% offset yield strength $\sigma_y$ in $\text{MPa}$.
- `tensile_strength_mpa_min`, `_max`, `_typ`: Ultimate tensile strength $\sigma_{uts}$ in $\text{MPa}$.
- `elongation_pct_typ`: Ductility / failure strain in percent (%).
- `fracture_toughness_mpam12_typ`: Mode-I plane-strain fracture toughness $K_{1c}$ in $\text{MPa}\sqrt{\text{m}}$.
- `hardness_hb_typ`: Brinell hardness number (HB).
- `thermal_conductivity_w_mk_typ`: Thermal conductivity $k$ in $\text{W/(m}\cdot\text{K)}$.
- `cost_usd_per_kg_typ`: Approximate raw stock material cost in $\text{USD/kg}$.
- `machinability_rating_pct`: Relative machinability index referenced to AISI 1212 cold rolled steel ($100\%$).
- `corrosion_resistance_score`: Coarse ordinal scale ($1 = \text{Poor/Atmospheric Rusting}$, $5 = \text{Excellent/Immune}$).
- `max_service_temp_c`: Maximum continuous service temperature in $^{\circ}\text{C}$.
- `source_reference`: Direct literature citation (ASM Handbooks, MMPDS, Machinery's Handbook, DIN EN standards, manufacturers' verified datasheets).

## 3. Provenance & Scientific Integrity Note
Data values represent standard engineering handbook figures. Property ranges reflect standard manufacturing and heat-treatment scatter. Cost and machinability figures are approximate market estimates and should be subjected to sensitivity analysis.
