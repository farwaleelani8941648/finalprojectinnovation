# Circulars AI Portal

**AI-Powered Classification & Semantic Search for Educational Circulars**  
_GitLab Repository_

---

## About the Project

This project is an end-to-end AI system for **automated classification and semantic retrieval of official circulars** (PDF documents) in educational or administrative contexts. It combines deep learning (sentence-transformer embeddings) with traditional machine learning (SVM/TFIDF fallback) for robust and accurate results. Users can upload, classify, browse, and search circulars through a user-friendly web interface.

**Author:** Umm E Farwa Leelani ([Farwa1648](https://gitlab.com/Farwa1648))  
**Supervisors:** Dauvin Emmanuel, Ramonfosse Laura, & Philippe Laurant

---

## Project Structure

project_inno/
│
├── app/ # FastAPI backend (ML logic, API)
│ ├── main.py
│ ├── predict.py
│ ├── search_engine.py
│ └── model/ # Trained models (.pkl)
│
├── circulars_admin/ # PDFs organized as MainCategory/SubCategory/
├── dataset/ # CSV datasets (auto-generated)
├── frontend/ # Flask web frontend (UI)
│ ├── app.py
│ ├── static/
│ └── templates/
├── scripts/ # Data processing & training scripts
│ ├── build_dataset.py
│ ├── clean_and_reset_dataset.py
│ ├── evaluate_classifier.py
│ ├── remove_test_circulars.py
│ ├── train_classifier.py
│ └── train_semantic_classifier.py
├── testings/ # Example/test PDFs (for demo & evaluation)
├── requirements.txt # Python dependencies
└── README.md


---

## Prerequisites

- **Python 3.8+** (tested with 3.10)
- **pip** ([install guide](https://pip.pypa.io/en/stable/))
- **Git** (for cloning the repository)
- **System dependencies** (for PDF extraction and semantic search):

    ```bash
    sudo apt update && sudo apt install poppler-utils libgl1
    ```

---

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://gitlab.com/Farwa1648/projectinnovation55983.git
    cd projectinnovation55983
    ```

2. **(Optional) Set up a Virtual Environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install Python Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Install the SpaCy French Model**
    ```bash
    python -m spacy download fr_core_news_sm
    ```

---

## Data Preparation & Model Training

**Step-by-step guide:**

1. **Organize Your PDFs**

    Place all your circular PDFs inside:

    ```
    circulars_admin/MainCategory/SubCategory/filename.pdf
    ```

    Example:

    ```
    circulars_admin/Staff_and_Personnel/Professional_Development/example.pdf
    ```

2. **Build the Dataset**
    ```bash
    python scripts/build_dataset.py
    ```

3. **(Optional) Clean and Reset the Dataset**
    ```bash
    python scripts/clean_and_reset_dataset.py
    ```

4. **(Optional) Remove Test/Dummy Circulars**
    ```bash
    python scripts/remove_test_circulars.py
    ```

5. **Train the SVM/TFIDF (Classic) Classifier**
    ```bash
    python scripts/train_classifier.py
    ```

6. **Train the Semantic Embedding Classifier**
    ```bash
    python scripts/train_semantic_classifier.py
    ```

7. **(Optional) Evaluate Classifier Accuracy**
    ```bash
    python scripts/evaluate_classifier.py
    ```

---

## Running the Application

### **Start the FastAPI Backend (Classification/Search API)**

From the project root:
```bash
uvicorn app.main:app --reload --port 8000

Or, if inside the app folder:

uvicorn main:app --reload --port 8000

Start the Flask Frontend (User Web Interface)

From the frontend directory:

cd frontend
python app.py

Access the site at: http://localhost:5000
Features

    Classification: Upload a PDF and the system auto-categorizes it using a hybrid deep/ML approach.

    Semantic Search: Find circulars by keywords, topics, or natural sentences (not just exact matches).

    Duplicate Detection: Identifies repeated circulars by number.

    Browsing: View all circulars by category and subcategory.

    Download: Download PDFs directly from the portal.

Project Report & Evaluation

    See the full technical report in Document 12.pdf in the project root.

    Classification Accuracy: 78% (on 80 real circulars)

    Hybrid ML: Sentence-transformer + SVM/TFIDF fallback

    Semantic Search: FAISS vector engine

    Demonstration & Test Cases: See testings/ folder

Requirements

    fastapi

    uvicorn

    flask

    pandas

    scikit-learn

    sentence-transformers

    faiss-cpu

    spacy

    PyMuPDF

    pdfplumber

    requests

    python-multipart

Note:
Install the SpaCy French model separately:

python -m spacy download fr_core_news_sm

Notes

    All scripts use script-relative paths; you can run them from anywhere.

    For best results, use machine-readable PDFs.

    If you add new circulars, re-run dataset scripts and retrain the models for highest accuracy.

    Demo scenarios: Use files in testings/ as shown in the report.

Credits

Author: Umm E Farwa Leelani (Farwa1648)
Supervisors: Dauvin Emmanuel, Ramonfosse Laura, & Philippe Laurant
Contact

For questions, issues, or feedback:

    Open an issue on GitLab

    Email: etu55983@henallux.be
