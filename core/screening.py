"""
Material Screening and Hard Constraints Filter Module.

Filters a materials database against design constraints (boundary conditions)
before multi-objective ranking or Pareto front generation.
"""

import pandas as pd
from typing import Dict, Any, Tuple, List, Optional


class MaterialScreener:
    """
    Applies boundary-condition filters and generates screening diagnostics.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def filter(
        self,
        min_yield_strength_mpa: Optional[float] = None,
        min_fracture_toughness: Optional[float] = None,
        min_elongation_pct: Optional[float] = None,
        min_youngs_modulus_gpa: Optional[float] = None,
        min_hardness_hb: Optional[float] = None,
        max_cost_usd_per_kg: Optional[float] = None,
        min_service_temp_c: Optional[float] = None,
        min_corrosion_score: Optional[int] = None,
        min_machinability_pct: Optional[float] = None,
        allowed_categories: Optional[List[str]] = None,
        excluded_categories: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Applies constraints and tracks exclusion statistics per constraint.
        """
        filtered = self.df.copy()
        rejection_stats = {}

        if allowed_categories is not None:
            mask = filtered["category"].isin(allowed_categories)
            rejection_stats["excluded_by_category"] = int((~mask).sum())
            filtered = filtered[mask]

        if excluded_categories is not None:
            mask = ~filtered["category"].isin(excluded_categories)
            rejection_stats["excluded_by_forbidden_category"] = int((~mask).sum())
            filtered = filtered[mask]

        if min_yield_strength_mpa is not None:
            mask = filtered["yield_strength_mpa_typ"] >= min_yield_strength_mpa
            rejection_stats["failed_yield_strength"] = int((~mask).sum())
            filtered = filtered[mask]

        if min_fracture_toughness is not None:
            mask = filtered["fracture_toughness_mpam12_typ"] >= min_fracture_toughness
            rejection_stats["failed_fracture_toughness"] = int((~mask).sum())
            filtered = filtered[mask]

        if min_elongation_pct is not None:
            mask = filtered["elongation_pct_typ"] >= min_elongation_pct
            rejection_stats["failed_ductility"] = int((~mask).sum())
            filtered = filtered[mask]

        if min_youngs_modulus_gpa is not None:
            mask = filtered["youngs_modulus_gpa_typ"] >= min_youngs_modulus_gpa
            rejection_stats["failed_youngs_modulus"] = int((~mask).sum())
            filtered = filtered[mask]

        if min_hardness_hb is not None:
            mask = filtered["hardness_hb_typ"] >= min_hardness_hb
            rejection_stats["failed_hardness"] = int((~mask).sum())
            filtered = filtered[mask]

        if max_cost_usd_per_kg is not None:
            mask = filtered["cost_usd_per_kg_typ"] <= max_cost_usd_per_kg
            rejection_stats["failed_max_cost"] = int((~mask).sum())
            filtered = filtered[mask]

        if min_service_temp_c is not None:
            mask = filtered["max_service_temp_c"] >= min_service_temp_c
            rejection_stats["failed_service_temp"] = int((~mask).sum())
            filtered = filtered[mask]

        if min_corrosion_score is not None:
            mask = filtered["corrosion_resistance_score"] >= min_corrosion_score
            rejection_stats["failed_corrosion_score"] = int((~mask).sum())
            filtered = filtered[mask]

        if min_machinability_pct is not None:
            mask = filtered["machinability_rating_pct"] >= min_machinability_pct
            rejection_stats["failed_machinability"] = int((~mask).sum())
            filtered = filtered[mask]

        return filtered, rejection_stats
