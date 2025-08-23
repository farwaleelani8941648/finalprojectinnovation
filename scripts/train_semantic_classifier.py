import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import pickle

# Get the directory where this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Use paths relative to the script location
DATASET_PATH = os.path.join(SCRIPT_DIR, "../dataset/circulars_dataset.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "../app/model")
SEMANTIC_CLF_PATH = os.path.join(MODEL_DIR, "semantic_classifier.pkl")
EMBEDDING_MODEL_PATH = os.path.join(MODEL_DIR, "embedding_model.pkl")

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Inputs and labels: USE RAW CONTENT (no cleaning/lemmatization)
X = df["content"]
y = df["subcategory"]

# Load multilingual sentence embedding model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Convert content to embeddings
print("🔄 Generating sentence embeddings (raw text)...")
X_embedded = model.encode(X.tolist(), show_progress_bar=True)

# Train classifier
print("🧠 Training logistic regression classifier...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_embedded, y)

# Save classifier and model
os.makedirs(MODEL_DIR, exist_ok=True)
with open(SEMANTIC_CLF_PATH, "wb") as f:
    pickle.dump(clf, f)
with open(EMBEDDING_MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("✅ Saved semantic model and classifier to app/model/")
