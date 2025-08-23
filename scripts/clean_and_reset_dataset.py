import os
import pandas as pd

# ========== PATH SETUP ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../dataset"))

AUGMENTED_DATASET_PATH = os.path.join(DATASET_DIR, "circulars_dataset_augmented.csv")
OFFICIAL_DATASET_PATH = os.path.join(DATASET_DIR, "circulars_dataset_augmented_official.csv")
NEW_CIRCULARS_PATH = os.path.join(DATASET_DIR, "new_circulars.csv")

# ========== STEP 1: LOAD DATA ==========
print("Loading augmented dataset from:", AUGMENTED_DATASET_PATH)
try:
    df = pd.read_csv(AUGMENTED_DATASET_PATH)
except pd.errors.EmptyDataError:
    print("ERROR: The dataset is empty or improperly formatted.")
    exit(1)
except FileNotFoundError:
    print("ERROR: Dataset file not found at path:", AUGMENTED_DATASET_PATH)
    exit(1)

# ========== STEP 2: CLEAN DATA ==========
df['main_category'] = df['main_category'].replace('Legal_and_Complaince', 'Legal_and_Compliance')

if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

df = df[df['content'].notnull()]
df = df[df['content'].str.strip() != '']
df['main_category'] = df['main_category'].str.strip()
df['subcategory'] = df['subcategory'].str.strip()

print(f"Rows after cleaning: {len(df)}")

# ========== STEP 3: OFFICIAL CIRCULAR NUMBERS ==========
official_circular_numbers = list(set([
    9215, 9355, 8936, 9289, 9348, 9487, 9333, 9212, 9397,
    9421, 9408, 9459, 9310, 9293, 9305, 9226, 9345, 9314,
    9197, 9104, 9286, 9335, 9341, 9271, 9339, 9358, 9478,
    9450, 9385, 9308, 9220, 9410, 9276, 9342, 9292, 9396,
    9387, 9441, 9377, 9407, 9451, 8742, 8882, 9388, 9217,
    8256, 8971, 9216, 9032, 9401, 9191, 9505, 9504, 9241,
    9418, 9477, 7137, 7177, 9427, 9244, 9454, 9496, 7679, 7616, 7887, 8200, 8113, 8479, 8568, 8912, 9002, 9005, 8980, 9331, 9515, 7594, 7742, 7869, 7971, 8052, 8160, 8737, 9229, 9347, 9440, 9009, 8708, 8526, 8624, 8655, 9279, 9516, 8645, 9307, 8984, 8884, 8976,8974, 9299, 9308, 9319, 9343, 8983,
 9300, 8976, 9549, 9390, 8797, 9306, 8795, 8608, 9106, 8878, 8762, 8473, 8653, 9489, 9425, 7602, 8539, 8349, 8077, 8425, 8532, 7796, 8360, 8409, 7602, 9472
, 7508, 7557, 9120, 8763, 9014, 7606, 7966, 7964, 7974, 8215]))

official_circular_numbers = [str(num) for num in official_circular_numbers]

# ========== STEP 4: FILTER DATA ==========
df_official = df[df["circular_number"].astype(str).isin(official_circular_numbers)]
df_new = df[~df["circular_number"].astype(str).isin(official_circular_numbers)]

print(f"Official records: {len(df_official)}")
print(f"New/unofficial records: {len(df_new)}")

# ========== STEP 5: SAVE OUTPUT ==========
os.makedirs(DATASET_DIR, exist_ok=True)

df_official.to_csv(OFFICIAL_DATASET_PATH, index=False)
print(f"Saved official dataset: {OFFICIAL_DATASET_PATH}")

if not df_new.empty:
    df_new.to_csv(NEW_CIRCULARS_PATH, index=False)
    print(f"Saved new circulars: {NEW_CIRCULARS_PATH}")

# ========== STEP 6: SHOW PREVIEW ==========
print("\n===== SAMPLE OF OFFICIAL DATASET =====")
print(df_official.head(5))

if not df_new.empty:
    print("\n===== SAMPLE OF NEW CIRCULARS =====")
    print(df_new.head(5))
