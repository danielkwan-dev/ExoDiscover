import pandas as pd

# Load your merged dataset
df = pd.read_csv("merged_exoplanets.csv")

# Keep only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Compute correlation matrix
corr = numeric_df.corr()

# Sort correlation of all features with the target label
print(corr['disposition'].sort_values(ascending=False))
