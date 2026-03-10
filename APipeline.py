import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestRegressor

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

# 7. Fit the Pipeline on the Training Data
print("Fitting pipeline to training data (this handles encoding, scaling, and training!)...")
pipeline.fit(X_train, y_train)

# 8. Evaluate on the Test Set
score = pipeline.score(X_test, y_test)
print(f"Pipeline successfully executed! Default Random Forest R^2 Score: {score:.4f}")