import pandas as pd

# 1. Load the original OHE dataset
df = pd.read_csv('SG_usedcar_ohe.csv')

# 2. Extract the brand OHE columns
brand_cols = [col for col in df.columns if col.startswith('Brand_')]
df_brands_ohe = df[brand_cols]

# 3. Find the column with a '1' and strip the 'Brand_' prefix to get the brand name
df['Brand'] = df_brands_ohe.idxmax(axis=1).str.replace('Brand_', '')

# 4. Drop the old OHE columns to clean up the dataset
df = df.drop(columns=brand_cols) 

# 5. Save the updated dataset to a new CSV file
df.to_csv('SG_usedcar.csv', index=False)

print("Successfully saved SG_usedcar.csv!")
print(df[['Brand', 'Price']].head())