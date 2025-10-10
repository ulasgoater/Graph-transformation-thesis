"""Model Training & Evaluation

This module trains machine learning models to predict whether a road segment
has pedestrian infrastructure (sidewalk) based on engineered spatial features.

Key concepts for beginners:
1. Features: Numeric or categorical values describing each road (length, speed, nearby amenities, etc.).
2. Label: 1 (has sidewalk) or 0 (no sidewalk) produced by distance-based logic.
3. Spatial Cross-Validation: We split the map into tiles to avoid training on
    a road and testing on an immediately adjacent road (reduces optimistic bias).
4. Random Forest: An ensemble of decision trees; robust and handles mixed
    feature types well.
5. Baselines: Simple models (logistic regression, distance-only rule) give a
    reference to show the Random Forest adds real value.
6. Out-of-Fold (OOF) Predictions: Probabilities predicted for each segment
    using only models that were NOT trained on that segment (fair evaluation).
"""

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import (
    precision_recall_curve, precision_score, recall_score,
    f1_score, average_precision_score, confusion_matrix, brier_score_loss
)

from config import N_SPLITS_WANTED, TILE_SIZE_M, RANDOM_STATE
from config import (
    MAX_FEATURES, RANDOM_STATE, CLASS_BALANCE_MIN_POSITIVE_RATIO, FEATURE_IMPORTANCES_CSV,
    ENABLE_BASELINE_LOGREG, PERMUTATION_IMPORTANCE_N_REPEATS, EXPORT_PERMUTATION_IMPORTANCE,
    PERMUTATION_IMPORTANCES_CSV
)


def train_and_evaluate(gdf, numeric_feats, categorical_feats,
                       n_splits_desired=N_SPLITS_WANTED, tile_size=TILE_SIZE_M):
    """
    Train and evaluate a Random Forest model using spatial cross-validation.
    
    Args:
        gdf: GeoDataFrame with features and labels
        numeric_feats: List of numeric feature column names
        categorical_feats: List of categorical feature column names
        n_splits_desired: Desired number of CV splits
        tile_size: Size of spatial tiles for grouping (meters)
    
    Returns:
        tuple: (gdf_with_predictions, preprocessor, classifier, metrics_dict)
    """
    # Ensure centroid exists
    if "centroid" not in gdf.columns:
        gdf = gdf.copy()
        gdf["centroid"] = gdf.geometry.centroid

    X = gdf[numeric_feats + categorical_feats].copy()
    y = gdf["has_ped"].astype(int).copy()

    # If only a single class present, we cannot train a discriminative model.
    # We create a trivial predictor (always that class) and return perfect or zero metrics accordingly.
    if y.nunique() < 2:
        single_class = int(y.iloc[0]) if len(y) else 0
        gdf_out = gdf.copy()
        gdf_out["pred_prob_oof"] = float(single_class)
        gdf_out["pred_label_oof"] = single_class
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if single_class == 1:
            # All positives: precision=1 (no false positives), recall=1, f1=1, AP=1
            precision_val = 1.0
            recall_val = 1.0
            f1_val = 1.0
            pr_auc_val = 1.0
        else:
            # All negatives: precision undefined (set to 0 by convention), recall=0, f1=0, AP=0
            precision_val = 0.0
            recall_val = 0.0
            f1_val = 0.0
            pr_auc_val = 0.0
        # Represent confusion matrix consistently as 2x2 even if one row/col is zero

        confusion = [[n_neg, 0], [0, n_pos]]
        metrics = {
            "precision": precision_val,
            "recall": recall_val,
            "f1": f1_val,
            "pr_auc": pr_auc_val,
            "brier": 0.0,  # constant perfect prediction
            "confusion_matrix": confusion,
            "opt_th": 0.5,
            "note": f"Only one class present (class={single_class}); metrics are trivial. pos={n_pos}, neg={n_neg}."
        }
        # Create a dummy preprocessor & classifier placeholders
        preproc = _create_preprocessor(numeric_feats, categorical_feats)
        clf = RandomForestClassifier(n_estimators=1, random_state=RANDOM_STATE)
        return gdf_out, preproc, clf, metrics

    # Compute spatial tile groups (tile id = coarse grid cell). These groups act like 'fold IDs'.
    gdf = gdf.copy()
    gdf["tx"] = (gdf.centroid.x // tile_size).astype(int)
    gdf["ty"] = (gdf.centroid.y // tile_size).astype(int)
    # Use an unambiguous tuple-ish string to avoid accidental collisions
    gdf["tile_id"] = gdf.apply(lambda r: f"({int(r['tx'])},{int(r['ty'])})", axis=1)
    groups = gdf["tile_id"]
    n_groups = groups.nunique()

    # Fallback if too few groups for spatial CV
    if n_groups < 2:
        return _fallback_stratified_split(X, y, numeric_feats, categorical_feats, gdf)

    # Spatial cross-validation
    return _spatial_cross_validation(X, y, groups, numeric_feats, categorical_feats, 
                                   gdf, n_splits_desired, n_groups)


def _fallback_stratified_split(X, y, numeric_feats, categorical_feats, gdf):
    """Fallback to stratified split when spatial CV is not possible."""
    warnings.warn("Too few spatial tiles for GroupKFold. Using stratified split.")  # fallback to traditional CV
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    
    # Create and fit preprocessor
    preproc = _create_preprocessor(numeric_feats, categorical_feats)
    preproc.fit(X_train)
    
    # Transform data
    Xtr = preproc.transform(X_train)
    Xte = preproc.transform(X_test)
    if hasattr(Xtr, "toarray"):
        Xtr = Xtr.toarray()
        Xte = Xte.toarray()
    
    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", 
        n_jobs=-1, random_state=RANDOM_STATE
    )
    clf.fit(Xtr, y_train)
    
    # Make predictions and evaluate
    probs = clf.predict_proba(Xte)[:, 1]
    opt_th = _find_optimal_threshold(y_test, probs)
    preds = (probs >= opt_th).astype(int)
    
    metrics = _calculate_metrics(y_test, preds, probs, opt_th)
    
    # Create output GeoDataFrame
    gdf_out = gdf.copy()
    gdf_out["pred_prob_oof"] = np.nan
    gdf_out.loc[X_test.index, "pred_prob_oof"] = probs
    gdf_out["pred_label_oof"] = (gdf_out["pred_prob_oof"] >= opt_th).astype("Int64")
    
    print("Fallback metrics:", metrics)
    return gdf_out, preproc, clf, metrics


def _spatial_cross_validation(X, y, groups, numeric_feats, categorical_feats, 
                            gdf, n_splits_desired, n_groups):
    """Perform spatial cross-validation using GroupKFold."""
    n_splits = min(n_splits_desired, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    
    # Create preprocessor and classifier
    preproc = _create_preprocessor(numeric_feats, categorical_feats)
    clf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", 
        n_jobs=-1, random_state=RANDOM_STATE
    )
    
    n = len(X)
    oof_probs = np.zeros(n)

    fold = 0
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        fold += 1
        Xtr, Xte = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        ytr = y.iloc[train_idx]

        # If the training fold has only one class, assign constant probability
        if ytr.nunique() < 2:
            const_prob = float(ytr.iloc[0])
            oof_probs[test_idx] = const_prob
            continue
        
    # Reduce rare categorical levels so validation folds don't see unseen categories frequently.
        # Handle rare categories for all categorical features (not just 'surface')
        for cat_col in [c for c in categorical_feats if c in Xtr.columns]:
            Xtr, Xte = _handle_rare_categories(Xtr, Xte, cat_col)
        
        # Fit preprocessor and transform data
        preproc.fit(Xtr)
        Xtr_t = preproc.transform(Xtr)
        Xte_t = preproc.transform(Xte)
        
        if hasattr(Xtr_t, "toarray"):
            Xtr_t = Xtr_t.toarray()
            Xte_t = Xte_t.toarray()
        
    # Train RandomForest on transformed training features and predict probabilities for held-out fold
        clf.fit(Xtr_t, ytr)
        probs = clf.predict_proba(Xte_t)
        if probs.shape[1] == 2:
            oof_probs[test_idx] = probs[:, 1]
        else:
            # Only one class learned in this fold after encoding; set constant probs
            learned_class = int(ytr.iloc[0])
            oof_probs[test_idx] = float(learned_class)

    # Calculate metrics on out-of-fold predictions
    opt_th = _find_optimal_threshold(y, oof_probs)
    preds_opt = (oof_probs >= opt_th).astype(int)
    metrics = _calculate_metrics(y, preds_opt, oof_probs, opt_th)
    
    print("Spatial CV metrics:", metrics)
    
    # Create output GeoDataFrame
    gdf_out = gdf.copy()
    gdf_out["pred_prob_oof"] = oof_probs
    gdf_out["pred_label_oof"] = (gdf_out["pred_prob_oof"] >= opt_th).astype(int)
    
    # After CV: optional baseline & permutation importance
    if ENABLE_BASELINE_LOGREG:
        try:
            base_preproc = _create_preprocessor(numeric_feats, categorical_feats)
            base_preproc.fit(X)
            Xt_all = base_preproc.transform(X)
            if hasattr(Xt_all, 'toarray'):
                Xt_all = Xt_all.toarray()
            logreg = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=None)
            logreg.fit(Xt_all, y)
            base_probs = logreg.predict_proba(Xt_all)[:,1]
            base_th = _find_optimal_threshold(y, base_probs)
            base_pred = (base_probs >= base_th).astype(int)
            base_metrics = _calculate_metrics(y, base_pred, base_probs, base_th)
            metrics['baseline_logreg'] = base_metrics
        except Exception as e:
            metrics['baseline_logreg_error'] = str(e)

    if EXPORT_PERMUTATION_IMPORTANCE:
        try:
            # Refit RF on all data
            full_preproc = _create_preprocessor(numeric_feats, categorical_feats)
            full_preproc.fit(X)
            Xt_all = full_preproc.transform(X)
            if hasattr(Xt_all, 'toarray'):
                Xt_all = Xt_all.toarray()
            rf_full = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)
            rf_full.fit(Xt_all, y)
            perm = permutation_importance(rf_full, Xt_all, y, n_repeats=PERMUTATION_IMPORTANCE_N_REPEATS, random_state=RANDOM_STATE, n_jobs=-1)
            # Build feature name list consistent with preprocessor
            num_names = numeric_feats
            # Extract OHE categories
            cat_encoder = full_preproc.named_transformers_['cat'].named_steps['onehot']
            cat_feature_names = list(cat_encoder.get_feature_names_out(full_preproc.transformers_[1][2]))
            all_feature_names = num_names + cat_feature_names
            perm_df = pd.DataFrame({
                'feature': all_feature_names,
                'mean_importance': perm.importances_mean,
                'std_importance': perm.importances_std
            }).sort_values('mean_importance', ascending=False)
            perm_df.to_csv(PERMUTATION_IMPORTANCES_CSV, index=False)
            metrics['permutation_importances_csv'] = PERMUTATION_IMPORTANCES_CSV
        except Exception as e:
            metrics['permutation_importances_error'] = str(e)

    # Distance-only baseline (if distance feature present)
    try:
        dist_col = 'ped_line_dist_m' if 'ped_line_dist_m' in X.columns else ('ped_dist_m' if 'ped_dist_m' in X.columns else None)
        if dist_col is not None:
            dist_vals = X[dist_col].astype(float)
            valid_mask = dist_vals.notna()
            # Simple threshold sweep (20 quantiles) optimizing F1 using only distance
            qs = np.linspace(0.05, 0.95, 20)
            best_f1 = -1
            best_th = None
            best_metrics = None
            d_valid = dist_vals[valid_mask]
            y_valid = y[valid_mask]
            if len(d_valid) > 0 and y_valid.nunique() == 2:
                for q in qs:
                    th = d_valid.quantile(q)
                    preds = (dist_vals <= th).astype(int)
                    f1v = f1_score(y, preds)
                    if f1v > best_f1:
                        m = _calculate_metrics(y, preds, 1 - (dist_vals / (dist_vals.max()+1e-9)).fillna(0), th)
                        best_f1 = f1v
                        best_th = th
                        best_metrics = m
                if best_metrics:
                    best_metrics['distance_feature'] = dist_col
                    metrics['baseline_distance_only'] = best_metrics
    except Exception as e:
        metrics['baseline_distance_only_error'] = str(e)

    return gdf_out, preproc, clf, metrics


def _create_preprocessor(numeric_feats, categorical_feats):
    """Create sklearn preprocessor pipeline."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), 
        ("scale", StandardScaler())
    ])
    # Compatibility across scikit-learn versions
    # If running on >=1.2, prefer 'sparse_output'; otherwise fall back to 'sparse'
    import sklearn
    ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if ver >= (1, 2):
        cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    else:
        cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse=True)
    
    preproc = ColumnTransformer([
        ("num", num_pipe, numeric_feats), 
        ("cat", cat_pipe, categorical_feats)
    ], sparse_threshold=0.3)
    
    return preproc


def _handle_rare_categories(Xtr, Xte, column):
    """Handle rare categories to prevent data leakage."""
    vc = Xtr[column].value_counts(normalize=True)
    common = vc[vc > 0.01].index.tolist()
    Xtr[column] = Xtr[column].where(Xtr[column].isin(common), other="other")
    Xte[column] = Xte[column].where(Xte[column].isin(common), other="other")
    return Xtr, Xte


def _find_optimal_threshold(y_true, y_probs):
    """Find optimal threshold based on F1 score."""
    prec, rec, thr = precision_recall_curve(y_true, y_probs)
    if len(thr) > 0:
        f1s = 2 * prec[1:] * rec[1:] / (prec[1:] + rec[1:] + 1e-12)
        opt_th = thr[int(np.nanargmax(f1s))]
    else:
        opt_th = 0.5
    return opt_th


def _calculate_metrics(y_true, y_pred, y_probs, threshold):
    """Calculate evaluation metrics including Brier score (probability calibration)."""
    try:
        brier = brier_score_loss(y_true, y_probs)
    except Exception:
        brier = None
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_probs),
        "brier": brier,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "opt_th": threshold
    }