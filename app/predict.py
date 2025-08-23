import pickle
import fitz  # PyMuPDF
import tempfile
import pandas as pd
import re
import spacy
import os
import math
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

# Load SpaCy French model (used for classic SVM preprocessing)
nlp = spacy.load('fr_core_news_sm')
nlp.max_length = 2_000_000

# Load dataset for circulars
dataset_path = "dataset/circulars_dataset.csv"
df = pd.read_csv(dataset_path)
df["circular_number"] = pd.to_numeric(df["circular_number"], errors='coerce')

# Load the trained semantic classifier and embedding model
with open("app/model/semantic_classifier.pkl", "rb") as f:
    semantic_clf: LogisticRegression = pickle.load(f)
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Paths for SVM/TFIDF fallback models (loaded on demand)
svm_vectorizer_path = "app/model/vectorizer.pkl"
svm_classifier_path = "app/model/classifier.pkl"

# Optional: mapping for known circulars that override classifier
manual_map = {
    9149: ("Staff_and_Personnel", "Professional_Development"),
    9185: ("Legal_and_Compliance", "Regulatory_Compliance"),
    9999: ("Legal_and_Compliance", "Reporting_obligations"),
}

def preprocess_text(text):
    """
    Basic text cleaning for SVM classifier.
    Tokenizes, lowercases, removes stopwords and non-alphabetic tokens, then lemmatizes.
    """
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

def extract_text_from_pdf_bytes(file_bytes):
    """
    Extracts all text from PDF file bytes using PyMuPDF.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            doc = fitz.open(tmp.name)
            return "".join(page.get_text() for page in doc).strip()
    except Exception as e:
        print(f"PDF error: {e}")
        return ""

def extract_circular_number(text_or_filename):
    """
    Extracts a 4-digit circular number from the given text or filename.
    Looks for patterns like (1234), [1234], or simply 1234.
    """
    match = re.search(r"[\(\[]\s*(\d{4})\s*[\)\]]", text_or_filename)
    if not match:
        match = re.search(r"\b(\d{4})\b", text_or_filename)
    return match.group(1) if match else None

def append_to_dataset(filename, main_category, subcategory, circular_number, file_bytes):
    """
    Appends new classification result and PDF content to the dataset.
    Also saves the PDF file to the categorized folder.
    """
    new_entry = {
        "filename": filename,
        "main_category": main_category,
        "subcategory": subcategory,
        "circular_number": circular_number or "0000",
        "content": extract_text_from_pdf_bytes(file_bytes)
    }
    df_existing = pd.read_csv(dataset_path)
    df_updated = pd.concat([df_existing, pd.DataFrame([new_entry])], ignore_index=True)
    df_updated.to_csv(dataset_path, index=False, quoting=1, escapechar='\\', encoding='utf-8')
    print(f"Appended new circular: {filename}")

    folder_path = os.path.join("circulars_admin", main_category, subcategory, "new")
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    print(f"Saved to: {file_path}")

def classify_uploaded_pdf(file_bytes, filename=None):
    """
    Main function to classify a PDF document:
    - Extracts text and circular number
    - Checks if document already exists in dataset
    - Runs semantic classifier (with manual override if needed)
    - Optionally falls back to SVM classifier on low confidence
    - Appends new entry to dataset and categorized folder
    Returns: Classification result dictionary.
    """
    text = extract_text_from_pdf_bytes(file_bytes)
    if not text.strip():
        print("Extracted text is empty!")
        return {"error": "No text extracted from PDF. Please check your file."}

    circular_number = extract_circular_number(filename or text)

    # Check for existing circular in dataset
    matches = []
    is_existing = False
    if circular_number:
        try:
            circular_number = int(circular_number.strip())
            matches = df[df["circular_number"] == circular_number].to_dict(orient="records")
            is_existing = len(matches) > 0
        except ValueError:
            pass

    if is_existing:
        print(f"Circular {circular_number} already exists with {len(matches)} match(es). Skipping prediction.")
        return {
            "main_category": None,
            "predicted_subcategory": None,
            "circular_number": circular_number,
            "existing": True,
            "matches": matches,
            "confidence": None
        }

    # Classification for new circular
    main_category = None
    predicted_sub = None
    confidence = None

    # Manual override if circular number is mapped
    if circular_number and int(circular_number) in manual_map:
        main_category, predicted_sub = manual_map[int(circular_number)]
        print("Manual override applied.")
        confidence = 1.0
    else:
        embedding = embedding_model.encode([text])
        try:
            probas = semantic_clf.predict_proba(embedding)[0]
            predicted_sub = semantic_clf.classes_[probas.argmax()]
            confidence = max(probas)
            print(f"Semantic classifier confidence: {confidence:.2f}")

            main_category_row = df[df["subcategory"] == predicted_sub]
            if not main_category_row.empty:
                main_category = main_category_row["main_category"].iloc[0]
        except NotFittedError:
            print("Model not fitted")
            return {
                "main_category": None,
                "predicted_subcategory": None,
                "circular_number": circular_number,
                "existing": False,
                "matches": [],
                "error": "Classifier not fitted",
                "confidence": None
            }

    # SVM classifier fallback for low-confidence predictions
    svm_pred = None
    svm_pred_confidence = None
    if confidence is not None and confidence < 0.5:
        print("Low confidence, running SVM classifier as fallback...")
        with open(svm_vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        with open(svm_classifier_path, "rb") as f:
            svm_clf = pickle.load(f)
        clean_text = preprocess_text(text)
        X_vec = vectorizer.transform([clean_text])
        svm_pred = svm_clf.predict(X_vec)[0]
        try:
            decision = svm_clf.decision_function(X_vec)
            svm_pred_confidence = float(decision.max())
        except Exception:
            svm_pred_confidence = None

    append_to_dataset(filename, main_category, predicted_sub, circular_number, file_bytes)

    # Ensure confidence is JSON-serializable
    safe_confidence = round(confidence, 3) if confidence is not None and math.isfinite(confidence) else None

    result = {
        "main_category": main_category,
        "predicted_subcategory": predicted_sub,
        "circular_number": circular_number,
        "existing": False,
        "matches": [],
        "confidence": safe_confidence
    }
    if svm_pred:
        result["svm_predicted_subcategory"] = svm_pred
        if svm_pred_confidence:
            result["svm_predicted_confidence"] = round(svm_pred_confidence, 3)

    print("Final confidence returned:", safe_confidence)
    return result
