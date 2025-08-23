import os
import pandas as pd

# Get directory where this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(SCRIPT_DIR, "../dataset/circulars_dataset.csv")

# List of test circular numbers to remove
test_circulars = [6553, 6554, 6555, 9149, 9188, 9440]

# Load dataset
df = pd.read_csv(dataset_path)

# Remove rows with these circular numbers
df_clean = df[~df['circular_number'].astype(str).isin([str(num) for num in test_circulars])]

# Save back to CSV
df_clean.to_csv(dataset_path, index=False)

print("Test circulars removed. Current dataset size:", len(df_clean))
