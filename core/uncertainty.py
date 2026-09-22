"""
Monte Carlo Uncertainty and Rank Stability Engine.

Propagates property tolerances (min/max/scatter) into index variations,
rank stability probabilities, and Pareto front membership confidence.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from core.ashby_indices import calculate_index
from core.pareto_optimizer import find_pareto_front


def run_monte_carlo_rank_stability(
    df: pd.DataFrame,
    index_key: str,
    n_iterations: int = 1000,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Performs Monte Carlo simulation over min/max property bounds.
    Evaluates how often each material ranks in Top 1, Top 3, and Top 5.
    """
    np.random.seed(random_seed)
    df_res = df.copy().reset_index(drop=True)
    num_materials = len(df_res)
    
    ranks_matrix = np.zeros((num_materials, n_iterations), dtype=int)
    index_values_matrix = np.zeros((num_materials, n_iterations), dtype=float)
    
    for it in range(n_iterations):
        # Sample uniformly between min and max
        sample_df = df_res.copy()
        
        # Density sample
        sample_df["density_sampled"] = np.random.uniform(
            sample_df["density_kg_m3_min"], sample_df["density_kg_m3_max"]
        )
        # Modulus sample
        sample_df["modulus_sampled"] = np.random.uniform(
            sample_df["youngs_modulus_gpa_min"], sample_df["youngs_modulus_gpa_max"]
        )
        # Yield sample
        sample_df["yield_sampled"] = np.random.uniform(
            sample_df["yield_strength_mpa_min"], sample_df["yield_strength_mpa_max"]
        )
        # Cost sample (assuming +-15% market volatility)
        cost_base = sample_df["cost_usd_per_kg_typ"].to_numpy()
        sample_df["cost_sampled"] = np.random.uniform(cost_base * 0.85, cost_base * 1.15)
        
        # Calculate sampled index
        idx_vals = calculate_index(
            sample_df,
            index_key,
            density_col="density_sampled",
            modulus_col="modulus_sampled",
            yield_col="yield_sampled",
            cost_col="cost_sampled"
        )
        
        index_values_matrix[:, it] = idx_vals.to_numpy()
        # Rank: 1 = highest index
        order = np.argsort(-idx_vals.to_numpy())
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, num_materials + 1)
        ranks_matrix[:, it] = ranks
        
    # Aggregate statistics
    prob_rank_1 = np.mean(ranks_matrix == 1, axis=1) * 100.0
    prob_top_3 = np.mean(ranks_matrix <= 3, axis=1) * 100.0
    prob_top_5 = np.mean(ranks_matrix <= 5, axis=1) * 100.0
    
    mean_rank = np.mean(ranks_matrix, axis=1)
    std_rank = np.std(ranks_matrix, axis=1)
    
    idx_p05 = np.percentile(index_values_matrix, 5, axis=1)
    idx_p50 = np.percentile(index_values_matrix, 50, axis=1)
    idx_p95 = np.percentile(index_values_matrix, 95, axis=1)
    
    df_res["mean_simulated_rank"] = mean_rank
    df_res["rank_std"] = std_rank
    df_res["prob_rank_1_pct"] = prob_rank_1
    df_res["prob_top_3_pct"] = prob_top_3
    df_res["prob_top_5_pct"] = prob_top_5
    df_res["index_p05"] = idx_p05
    df_res["index_median"] = idx_p50
    df_res["index_p95"] = idx_p95
    
    return df_res.sort_values(by="prob_top_3_pct", ascending=False)
