"""Pareto ranking and TOPSIS scoring for multi-objective material selection."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional


def find_pareto_front(
    df: pd.DataFrame,
    objectives: List[str],
    directions: List[str]  # 'min' or 'max' for each objective
) -> pd.DataFrame:
    """
    X dominates Y iff X is at least as good on every objective and strictly better on one.
    Adds a 'pareto_rank' column (1 = non-dominated frontier).
    """
    if len(objectives) != len(directions):
        raise ValueError("Length of objectives and directions must match.")

    df_res = df.copy()
    num_items = len(df_res)

    # Flip 'max' objectives so every column can be compared as a minimization
    obj_matrix = np.zeros((num_items, len(objectives)))
    for idx, (col, direction) in enumerate(zip(objectives, directions)):
        vals = df_res[col].to_numpy(dtype=float)
        if direction == "max":
            obj_matrix[:, idx] = -vals
        elif direction == "min":
            obj_matrix[:, idx] = vals
        else:
            raise ValueError(f"Direction must be 'min' or 'max', got: {direction}")

    domination_counts = np.zeros(num_items, dtype=int)
    dominated_sets = [[] for _ in range(num_items)]
    ranks = np.zeros(num_items, dtype=int)
    
    for p in range(num_items):
        for q in range(num_items):
            if p == q:
                continue
            p_better_or_equal = np.all(obj_matrix[p] <= obj_matrix[q])
            p_strictly_better = np.any(obj_matrix[p] < obj_matrix[q])
            
            if p_better_or_equal and p_strictly_better:
                dominated_sets[p].append(q)
            elif np.all(obj_matrix[q] <= obj_matrix[p]) and np.any(obj_matrix[q] < obj_matrix[p]):
                domination_counts[p] += 1
                
    current_front = np.where(domination_counts == 0)[0]
    current_rank = 1
    
    while len(current_front) > 0:
        next_front = []
        for p in current_front:
            ranks[p] = current_rank
            for q in dominated_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        current_rank += 1
        current_front = np.array(next_front, dtype=int)
        
    df_res["pareto_rank"] = ranks
    df_res["is_pareto_optimal"] = (ranks == 1)
    return df_res.sort_values(by="pareto_rank", ascending=True)


def calculate_topsis_score(
    df: pd.DataFrame,
    objectives: List[str],
    directions: List[str],
    weights: Optional[List[float]] = None
) -> pd.DataFrame:
    """TOPSIS: ranks materials by relative closeness to the ideal solution (least distance
    from the best-case vector, most distance from the worst-case vector)."""
    df_res = df.copy()
    if weights is None:
        weights = [1.0 / len(objectives)] * len(objectives)
    weights = np.array(weights) / np.sum(weights)

    matrix = df_res[objectives].to_numpy(dtype=float)

    denom = np.sqrt(np.sum(matrix ** 2, axis=0))
    denom[denom == 0] = 1.0
    norm_matrix = matrix / denom
    weighted_matrix = norm_matrix * weights

    ideal_best = np.zeros(len(objectives))
    ideal_worst = np.zeros(len(objectives))
    
    for j, direction in enumerate(directions):
        if direction == "max":
            ideal_best[j] = np.max(weighted_matrix[:, j])
            ideal_worst[j] = np.min(weighted_matrix[:, j])
        else:
            ideal_best[j] = np.min(weighted_matrix[:, j])
            ideal_worst[j] = np.max(weighted_matrix[:, j])

    d_pos = np.sqrt(np.sum((weighted_matrix - ideal_best) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weighted_matrix - ideal_worst) ** 2, axis=1))
    topsis_score = d_neg / (d_pos + d_neg + 1e-12)
    
    df_res["topsis_score"] = topsis_score
    return df_res.sort_values(by="topsis_score", ascending=False)
