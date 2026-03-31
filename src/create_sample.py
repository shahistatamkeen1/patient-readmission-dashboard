import pandas as pd
import os

os.makedirs("assets", exist_ok=True)

df = pd.read_csv("data/processed/cleaned_data.csv")
sample_df = df.sample(3000, random_state=42)
sample_df.to_csv("assets/sample_cleaned_data.csv", index=False)

print("Sample file created successfully!")
print("Saved at: assets/sample_cleaned_data.csv")
print("Shape:", sample_df.shape)