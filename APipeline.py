import pandas as pd
from datetime import datetime

# Scikit-Learn tools
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, root_mean_squared_error

# Encoders
from category_encoders import TargetEncoder

# Models (The 3 Families + Champion Candidate)
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

def main():
    # 1. Load your dataset
    print("Loading data...")
    df = pd.read_csv('SG_usedcar_Reversed.csv')

    # 2. Engineer AGE from Manufacturing date
    current_year = datetime.now().year
    df['Age'] = current_year - df['Manufactured']

    # 3. Separate features (X) and target (y)
    # We drop 'Price' (Target) and 'Manufactured' (since we engineered 'Age' from it)
    X = df.drop(columns=['Price', 'Manufactured'])
    y = df['Price']

    # 4. Train/Test Split (Strictly before the pipeline to prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 5. Define your feature branches
    brand_col = ['Brand']
    num_cols = ['Coe_left', 'Mileage', 'Road Tax', 'COE', 'Engine Cap', 
                'Curb Weight', 'Age', 'OMV', 'Power', 'No. of Owners']
    # Note: Type_* and Transmission_* are handled automatically by 'remainder=passthrough'

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
    # PART B: MODEL SELECTION (Execution)
    # ==========================================

    models = {
        "Linear Family (Ridge)": Ridge(),
        "Distance Family (KNN)": KNeighborsRegressor(),
        "Tree Ensemble (Random Forest)": RandomForestRegressor(random_state=42, n_jobs=-1),
        "Tree Ensemble (LightGBM)": LGBMRegressor(random_state=42, verbose=-1)
    }

    # Define the custom scorer for scikit-learn
    rmse_scorer = make_scorer(root_mean_squared_error)

    print("Starting Model Evaluation (5-Fold Cross Validation)...")
    print("This may take a minute depending on your CPU.\n")
    print("=" * 45)

    # 7. Evaluate each model through the pipeline
    for name, model in models.items():
        # Assemble the full pipeline for the current model
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('estimator', model)
        ])
        
        # Calculate scores using 5-fold cross-validation
        # n_jobs=-1 tells scikit-learn to use all available CPU cores for speed
        r2 = cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        rmse = cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring=rmse_scorer, n_jobs=-1)
        
        # Print results
        print(f"{name}")
        print(f"  R²  : {r2.mean():.4f} ± {r2.std():.4f}")
        print(f"  RMSE: ${rmse.mean():,.2f} ± ${rmse.std():,.2f}")
        print("-" * 45)

if __name__ == "__main__":
    main()