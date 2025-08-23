import pandas as pd
import pickle
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("dataset/circulars_dataset.csv")

# Input (text) and true labels
X = df["content"]
y_true = df["subcategory"]

# Load vectorizer and classifier
with open("app/model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("app/model/classifier.pkl", "rb") as f:
    clf = pickle.load(f)

# Transform dataset using the vectorizer
X_vec = vectorizer.transform(X)

# Predict using classifier
y_pred = clf.predict(X_vec)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_true, y_pred))

print("Overall Accuracy:", accuracy_score(y_true, y_pred))
