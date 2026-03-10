import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, root_mean_squared_error
# 1. For the Linear Family
from sklearn.linear_model import Ridge

# 2. For the Distance Family
from sklearn.neighbors import KNeighborsRegressor

# 3. For the Tree Ensemble Family
from sklearn.ensemble import RandomForestRegressor

# 4. For evaluating the pipeline
from sklearn.model_selection import cross_val_score



# 1. Load your dataset (Make sure this is the file with the single 'Brand' column!)
print("Loading data...")
df = pd.read_csv('SG_usedcar_Reversed.csv')

# 2. Separate features (X) and target (y)
X = df.drop(columns=['Price'])
y = df['Price']

# 3. Train/Test Split (CRITICAL to do this before the pipeline)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Define your feature branches
brand_col = ['Brand']
num_cols = ['Coe_left', 'Mileage', 'Road Tax', 'COE', 'Engine Cap', 
            'Curb Weight', 'Manufactured', 'OMV', 'Power', 'No. of Owners']
# Type_* and Transmission_* are left out because 'remainder=passthrough' handles them automatically

# 5. Build the Preprocessing Architecture
print("Building pipeline architecture...")
preprocessor = ColumnTransformer(
    transformers=[
        ('brand_encoder', TargetEncoder(smoothing=5), brand_col),
        ('num_scaler', StandardScaler(), num_cols)
    ],
    remainder='passthrough' # Passes your 0/1 binary columns through untouched
)

# 6. Assemble the Final Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('estimator', RandomForestRegressor(random_state=42, n_jobs=-1))
])


# ==========================================
# PART B: MODEL SELECTION (Execution)
# ==========================================

models = {
    "Linear Family (Ridge)": Ridge(),
    "Distance Family (KNN)": KNeighborsRegressor(),
    "Tree Ensemble (Random Forest)": RandomForestRegressor(random_state=42, n_jobs=-1)
}

rmse_scorer = make_scorer(root_mean_squared_error)

for name, model in models.items():
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('estimator', model)
    ])
    
    r2 = cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring='r2')
    rmse = cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring=rmse_scorer)
    
    print(f"{name}")
    print(f"  R²  : {r2.mean():.4f} ± {r2.std():.4f}")
    print(f"  RMSE: {rmse.mean():.2f} ± {rmse.std():.2f}")
    print()