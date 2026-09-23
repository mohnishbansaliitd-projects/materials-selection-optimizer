"""
Fills missing properties with physical relations first, then benchmarks ML imputers
against them using leave-one-category-out CV so no alloy family leaks into its own fit.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score


def physics_informed_imputation(df: pd.DataFrame) -> pd.DataFrame:
    df_imputed = df.copy()

    # Isotropic elasticity: G = E / [2(1 + nu)]
    mask_g_missing = df_imputed["shear_modulus_gpa_typ"].isna()
    if mask_g_missing.any():
        E = df_imputed.loc[mask_g_missing, "youngs_modulus_gpa_typ"]
        nu = df_imputed.loc[mask_g_missing, "poissons_ratio_typ"].fillna(0.30)
        df_imputed.loc[mask_g_missing, "shear_modulus_gpa_typ"] = E / (2.0 * (1.0 + nu))

    # Tabor's relation for ductile metals: HB ~ UTS / 3.45
    mask_hb_missing = df_imputed["hardness_hb_typ"].isna()
    if mask_hb_missing.any():
        uts = df_imputed.loc[mask_hb_missing, "tensile_strength_mpa_typ"]
        df_imputed.loc[mask_hb_missing, "hardness_hb_typ"] = uts / 3.45

    return df_imputed


def evaluate_grouped_imputation(
    df: pd.DataFrame,
    target_col: str = "yield_strength_mpa_typ",
    feature_cols: Optional[list] = None
) -> Dict[str, float]:
    """Leave-One-Group-Out by category, so a model is never tested on the family it trained on."""
    if feature_cols is None:
        feature_cols = ["density_kg_m3_typ", "youngs_modulus_gpa_typ", "hardness_hb_typ"]
        
    clean_df = df.dropna(subset=feature_cols + [target_col, "category"]).copy()
    X = clean_df[feature_cols].to_numpy()
    y = clean_df[target_col].to_numpy()
    groups = clean_df["category"].to_numpy()
    
    logo = LeaveOneGroupOut()
    y_preds_knn = []
    y_preds_rf = []
    y_trues = []
    
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    knn = KNeighborsRegressor(n_neighbors=3)
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        rf.fit(X_train, y_train)
        knn.fit(X_train, y_train)
        
        y_preds_rf.extend(rf.predict(X_test))
        y_preds_knn.extend(knn.predict(X_test))
        y_trues.extend(y_test)
        
    mae_rf = mean_absolute_error(y_trues, y_preds_rf)
    r2_rf = r2_score(y_trues, y_preds_rf)
    mae_knn = mean_absolute_error(y_trues, y_preds_knn)
    r2_knn = r2_score(y_trues, y_preds_knn)
    
    return {
        "rf_grouped_mae": float(mae_rf),
        "rf_grouped_r2": float(r2_rf),
        "knn_grouped_mae": float(mae_knn),
        "knn_grouped_r2": float(r2_knn),
        "n_samples": len(y_trues)
    }
