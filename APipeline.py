import pandas as pd
import numpy as np
from datetime import datetime

# Scikit-Learn tools
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, root_mean_squared_error

# Encoders
from category_encoders import TargetEncoder

# Models
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings('ignore')

def cv_rmse_dollars(pipeline, X, y_log, cv=5):
    """Compute RMSE in dollar scale across CV folds."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    fold_rmses = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_pred_dollars = np.expm1(pipeline.predict(X_val))
        y_val_dollars  = np.expm1(y_val)
        fold_rmses.append(root_mean_squared_error(y_val_dollars, y_pred_dollars))

    return np.array(fold_rmses)

def main():
    # 1. Load your dataset
    print("Loading data...")
    df = pd.read_csv('SG_usedcar_Reversed.csv')

    # 2. Engineer AGE from Manufacturing date
    current_year = datetime.now().year
    df['Age'] = current_year - df['Manufactured']

    # ==========================================
    # SLIDE 22: TUNING / IMPROVEMENTS
    # ==========================================

    # --- Improvement 1: Feature Engineering ---
    df['OMV_per_cc']        = df['OMV'] / df['Engine Cap']
    df['Power_per_kg']      = df['Power'] / df['Curb Weight']
    df['Mileage_per_year']  = df['Mileage'] / df['Age'].replace(0, 1)
    df['COE_remaining_pct'] = df['Coe_left'] / 3650  # % of 10yr COE left

    # 3. Separate features (X) and target (y)
    X = df.drop(columns=['Price', 'Manufactured'])
    y = df['Price']

    # --- Improvement 2: Log-transform the target (car prices are right-skewed) ---
    y_log = np.log1p(y)

    # 4. Train/Test Split
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.3, random_state=42
    )
    # Keep original scale y_test for final RMSE reporting in dollars
    _, _, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 5. Define your feature branches (now includes engineered features)
    brand_col = ['Brand']
    num_cols = ['Coe_left', 'Mileage', 'Road Tax', 'COE', 'Engine Cap',
                'Curb Weight', 'Age', 'OMV', 'Power', 'No. of Owners',
                'OMV_per_cc', 'Power_per_kg', 'Mileage_per_year', 'COE_remaining_pct']

    # 6. Build the Preprocessing Architecture
    print("Building pipeline architecture...\n")
    preprocessor = ColumnTransformer(
        transformers=[
            ('brand_encoder', TargetEncoder(smoothing=5), brand_col),
            ('num_scaler', StandardScaler(), num_cols)
        ],
        remainder='passthrough'
    )

    # ==========================================
    # PART B: MODEL SELECTION (Baseline)
    # ==========================================

    models = {
        "Linear Family (Ridge)": Ridge(),
        "Distance Family (KNN)": KNeighborsRegressor(),
        "Tree Ensemble (Random Forest)": RandomForestRegressor(random_state=42, n_jobs=-1),
        "Tree Ensemble (LightGBM)": LGBMRegressor(random_state=42, verbose=-1)
    }

    rmse_scorer = make_scorer(root_mean_squared_error)

    print("Starting Baseline Model Evaluation (5-Fold Cross Validation)...")
    print("=" * 45)

    for name, model in models.items():
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('estimator', model)
        ])

        r2           = cross_val_score(full_pipeline, X_train, y_train_log, cv=5, scoring='r2', n_jobs=-1)
        rmse         = cross_val_score(full_pipeline, X_train, y_train_log, cv=5, scoring=rmse_scorer, n_jobs=-1)
        rmse_dollars = cv_rmse_dollars(full_pipeline, X_train, y_train_log)

        print(f"{name}")
        print(f"  R²  : {r2.mean():.4f} ± {r2.std():.4f}")
        print(f"  RMSE (log-scale): {rmse.mean():.4f} ± {rmse.std():.4f}")
        print(f"  RMSE (dollars)  : ${rmse_dollars.mean():,.2f} ± ${rmse_dollars.std():,.2f}")
        print("-" * 45)

    # ==========================================
    # SLIDE 22: TUNING / IMPROVEMENTS (Cont.)
    # ==========================================

    # --- Improvement 3: Hyperparameter Tuning on LightGBM ---
    print("\n[Slide 22] Tuning LightGBM with RandomizedSearchCV...")
    print("=" * 45)

    param_grid = {
        'estimator__n_estimators':     [300, 500, 1000],
        'estimator__learning_rate':    [0.01, 0.05, 0.1],
        'estimator__max_depth':        [4, 6, 8],
        'estimator__num_leaves':       [31, 63, 127],
        'estimator__subsample':        [0.7, 0.8, 1.0],
        'estimator__colsample_bytree': [0.7, 0.8, 1.0],
    }

    lgbm_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('estimator', LGBMRegressor(random_state=42, verbose=-1))
    ])

    search = RandomizedSearchCV(
        lgbm_pipeline, param_grid,
        n_iter=30, cv=5, scoring='r2',
        random_state=42, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train_log)

    print(f"\nBest R² (tuned LightGBM): {search.best_score_:.4f}")
    print(f"Best Params: {search.best_params_}")

    # --- Improvement 4: Stacking Ensemble ---
    print("\n[Slide 22] Evaluating Stacking Ensemble...")
    print("=" * 45)

    stack_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('estimator', StackingRegressor(
            estimators=[
                ('rf',   RandomForestRegressor(random_state=42, n_jobs=-1)),
                ('lgbm', LGBMRegressor(random_state=42, verbose=-1)),
                ('knn',  KNeighborsRegressor())
            ],
            final_estimator=Ridge(),  # Meta-learner combines all 3
            cv=5
        ))
    ])

    r2_stack         = cross_val_score(stack_pipeline, X_train, y_train_log, cv=5, scoring='r2', n_jobs=-1)
    rmse_stack       = cross_val_score(stack_pipeline, X_train, y_train_log, cv=5, scoring=rmse_scorer, n_jobs=-1)
    rmse_stack_dollars = cv_rmse_dollars(stack_pipeline, X_train, y_train_log)

    print("Stacking Ensemble")
    print(f"  R²  : {r2_stack.mean():.4f} ± {r2_stack.std():.4f}")
    print(f"  RMSE (log-scale): {rmse_stack.mean():.4f} ± {rmse_stack.std():.4f}")
    print(f"  RMSE (dollars)  : ${rmse_stack_dollars.mean():,.2f} ± ${rmse_stack_dollars.std():,.2f}")
    print("-" * 45)

    # --- Summary ---
    print("\n[Slide 22] Improvement Summary")
    print("=" * 45)
    print(f"  Baseline LightGBM R²  : (see above)")
    print(f"  Tuned LightGBM R²     : {search.best_score_:.4f}")
    print(f"  Stacking Ensemble R²  : {r2_stack.mean():.4f}")

    # --- Dollar-scale Final Evaluation ---
    print("\n[Slide 22] Final Evaluation in Dollars")
    print("=" * 45)

    # Fit the tuned LightGBM on full training set
    best_pipeline = search.best_estimator_
    best_pipeline.fit(X_train, y_train_log)

    # Predict and inverse-transform back to dollars
    y_pred_log     = best_pipeline.predict(X_test)
    y_pred_dollars = np.expm1(y_pred_log)
    y_test_dollars = np.expm1(y_test_log)

    # Calculate final metrics in dollars
    from sklearn.metrics import r2_score
    final_r2   = r2_score(y_test_dollars, y_pred_dollars)
    final_rmse = root_mean_squared_error(y_test_dollars, y_pred_dollars)

    print(f"  Tuned LightGBM (Test Set)")
    print(f"  R²  : {final_r2:.4f}")
    print(f"  RMSE: ${final_rmse:,.2f}")
    print("-" * 45)

if __name__ == "__main__":
    main()