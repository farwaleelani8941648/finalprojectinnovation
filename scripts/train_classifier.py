import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle
import spacy

# Load SpaCy French model for NLP preprocessing
nlp = spacy.load('fr_core_news_sm')
nlp.max_length = 2000000

# Get the directory where this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Use paths relative to the script location
DATASET_PATH = os.path.join(SCRIPT_DIR, "../dataset/circulars_dataset.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "../app/model")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.pkl")

# Text preprocessing function
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Apply preprocessing
print("⏳ Preprocessing text data...")
df["clean_content"] = df["content"].apply(preprocess_text)

# Input (preprocessed text) and Output (subcategory)
X = df["clean_content"]
y = df["subcategory"]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)

# Fit and transform the preprocessed text
X_vec = vectorizer.fit_transform(X)

# Train Linear SVC classifier
clf = LinearSVC()
clf.fit(X_vec, y)

# Save models
os.makedirs(MODEL_DIR, exist_ok=True)

with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

with open(CLASSIFIER_PATH, "wb") as f:
    pickle.dump(clf, f)

print("✅ Enhanced classifier trained and saved to app/model/")
