# Feature Selection Code Improvements

## Improved Version of Cell 5.5

Here's an improved version of the feature selection code with better error handling and a helper function:

```python
# ======================================================
# Cell 5.5: Feature Selection Utilities (Improved)
# ======================================================

def select_features(X_train, y_train, X_val, y_val, method='mutual_info', k=100):
    """
    Select top k features using various methods.
    
    Args:
        X_train: Training features (DataFrame)
        y_train: Training labels
        X_val: Validation features (DataFrame, for RFE)
        y_val: Validation labels (for RFE)
        method: 'mutual_info', 'f_classif', or 'rfe'
        k: Number of features to select
    
    Returns:
        selected_features: List of selected feature names
        selector: Fitted selector object (for transforming test data)
    """
    # Ensure k is not larger than available features
    k = min(k, X_train.shape[1])
    
    if k <= 0:
        raise ValueError(f"Invalid k={k}. Must be positive and <= {X_train.shape[1]}")
    
    if method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()].tolist()
        return selected_features, selector
    
    elif method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()].tolist()
        return selected_features, selector
    
    elif method == 'rfe':
        # Use Random Forest as base estimator for RFE
        base_estimator = RandomForestClassifier(
            n_estimators=50,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=N_JOBS
        )
        selector = RFE(
            estimator=base_estimator,
            n_features_to_select=k,
            step=max(1, X_train.shape[1] // 20)  # Remove 5% features at a time
        )
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()].tolist()
        return selected_features, selector
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: 'mutual_info', 'f_classif', 'rfe'")

def apply_feature_selection(X_train, X_val, X_test, selected_features):
    """
    Apply feature selection to train, validation, and test sets.
    
    Args:
        X_train: Training features (DataFrame)
        X_val: Validation features (DataFrame)
        X_test: Test features (DataFrame)
        selected_features: List of selected feature names
    
    Returns:
        X_train_selected, X_val_selected, X_test_selected
    """
    # Ensure all selected features exist in all datasets
    available_features = set(X_train.columns) & set(X_val.columns) & set(X_test.columns)
    selected_features = [f for f in selected_features if f in available_features]
    
    if len(selected_features) == 0:
        raise ValueError("No common features found across train/val/test sets")
    
    if len(selected_features) < len([f for f in selected_features if f in X_train.columns]):
        print(f"Warning: {len(selected_features)} features available, some may be missing")
    
    X_train_selected = X_train[selected_features].copy()
    X_val_selected = X_val[selected_features].copy()
    X_test_selected = X_test[selected_features].copy()
    
    return X_train_selected, X_val_selected, X_test_selected

# Feature selection configuration
USE_FEATURE_SELECTION = False  # Set to True to enable feature selection
FEATURE_SELECTION_METHOD = 'mutual_info'  # Options: 'mutual_info', 'f_classif', 'rfe'
N_SELECTED_FEATURES = 100  # Number of features to select

# Store feature selectors per pollutant-horizon combination
feature_selectors = {}  # Key: (pollutant, horizon), Value: (selected_features, selector)

print("=" * 80)
print("Feature Selection Utilities")
print("=" * 80)
print(f"Feature selection enabled: {USE_FEATURE_SELECTION}")
if USE_FEATURE_SELECTION:
    print(f"Method: {FEATURE_SELECTION_METHOD}")
    print(f"Number of features to select: {N_SELECTED_FEATURES}")
    print(f"Original number of features: {X_tr.shape[1]}")
    print(f"Reduction: {X_tr.shape[1] - N_SELECTED_FEATURES} features will be removed")
else:
    print("All features will be used (no feature selection)")
print("=" * 80)
```

## Key Improvements

1. **Added `apply_feature_selection()` helper function**: 
   - Ensures selected features exist in all datasets
   - Handles missing features gracefully
   - Returns properly filtered DataFrames

2. **Better error handling**:
   - Validates `k` parameter
   - Checks for common features across datasets
   - Provides informative error messages

3. **Feature selector storage**:
   - `feature_selectors` dictionary to store selectors per pollutant-horizon
   - Useful for later analysis or applying to new data

## How to Use in Model Training

Add this code block at the beginning of each model training loop (in Cells 7-10):

```python
# ===== FEATURE SELECTION (if enabled) =====
if USE_FEATURE_SELECTION:
    # Select features using training data
    selected_features, selector = select_features(
        X_train, y_train, X_val, y_val,
        method=FEATURE_SELECTION_METHOD,
        k=N_SELECTED_FEATURES
    )
    
    # Store selector for this pollutant-horizon combination
    feature_selectors[(pollutant, h)] = (selected_features, selector)
    
    # Apply feature selection to all datasets
    X_train, X_val, X_test = apply_feature_selection(
        X_train, X_val, X_test, selected_features
    )
    
    print(f"  Selected {len(selected_features)} features for {pollutant} - {h}h")
# ===== END FEATURE SELECTION =====
```

## Testing Feature Selection

To test feature selection:

1. **Enable feature selection** in Cell 5.5:
   ```python
   USE_FEATURE_SELECTION = True
   FEATURE_SELECTION_METHOD = 'mutual_info'
   N_SELECTED_FEATURES = 100
   ```

2. **Add the feature selection code** to one model training cell (e.g., Cell 7 for Linear Models)

3. **Compare results** with and without feature selection

4. **Try different methods**:
   - `'mutual_info'`: Fast, good for non-linear relationships
   - `'f_classif'`: Fast, good for linear relationships
   - `'rfe'`: Slower but more thorough, uses model-based selection

## Expected Benefits

- **Reduced overfitting**: Fewer features = less overfitting risk
- **Faster training**: Smaller feature space = faster model training
- **Better interpretability**: Focus on most important features
- **Potential performance improvement**: Removing noise features can improve F1 by 2-5%

## Notes

- Feature selection is performed **per pollutant-horizon combination**
- Different features may be selected for different horizons
- Feature selection uses only training data (no data leakage)
- Selected features are stored in `feature_selectors` dictionary for later use





