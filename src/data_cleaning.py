import pandas as pd
import numpy as np
import os

# Create processed folder if it does not exist
os.makedirs("data/processed", exist_ok=True)

# Load dataset
df = pd.read_csv("data/raw/diabetic_data.csv")

print("Original Shape:", df.shape)

# Step 1: Remove unnecessary columns
drop_cols = ["encounter_id", "patient_nbr"]
df = df.drop(columns=drop_cols)

# Step 2: Replace ? with NaN
df = df.replace("?", np.nan)

# Step 3: Show missing values before cleaning
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# Step 4: Fill missing values
for col in df.select_dtypes(include=["int64", "float64"]).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Step 5: Create target column
df["target_readmit_30"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

# Drop old target column
df = df.drop(columns=["readmitted"])

# Step 6: Convert text columns into numbers
df = pd.get_dummies(df, drop_first=True)

# Step 7: Save cleaned data
output_path = "data/processed/cleaned_data.csv"
df.to_csv(output_path, index=False)

print("\nFinal Shape:", df.shape)
print(f"\nCleaned data saved successfully at: {output_path}")

df = df.sample(20000, random_state=42)