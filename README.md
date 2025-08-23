# Circulars AI Portal

**AI-Powered Classification & Semantic Search for Educational Circulars**

> End-to-end system to upload, classify, browse, and semantically search official PDF circulars.

**Author:** Umm E Farwa Leelani ([@Farwa1648](https://gitlab.com/Farwa1648))  
**Supervisors:** Dauvin Emmanuel, Ramonfosse Laura, Philippe Laurent

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Data Preparation & Training](#data-preparation--training)
- [Run the Apps](#run-the-apps)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Credits](#credits)
- [Contact](#contact)
- [License](#license)

---

## Overview
This project provides an **AI pipeline** that:
- extracts text from PDFs,
- detects the circular number (for deduplication/traceability),
- classifies each document into a **fixed taxonomy** (Main Category → Subcategory),
- computes a confidence score,
- and **falls back** to a classic TF-IDF/SVM model when semantic confidence is low.

A minimal **Flask frontend** lets users upload/search, while a **FastAPI backend** exposes ML endpoints used by the UI.

---

## Features
- **Hybrid ML:** sentence-transformer embeddings + TF-IDF/SVM fallback  
- **Semantic search:** query by meaning, not just keywords  
- **Duplicate detection:** via 4-digit circular numbers  
- **Browse by taxonomy:** main category & subcategory pages  
- **Simple web UI:** Flask app for demo and evaluation

---

## Tech Stack
- **Backend API:** FastAPI, Uvicorn  
- **Frontend:** Flask, Jinja2, static assets  
- **ML / NLP:** scikit-learn, sentence-transformers, spaCy (fr), pandas  
- **PDF:** PyMuPDF (`fitz`), pdfplumber  
- **Semantic Index (optional):** FAISS (CPU)

---

## Repository Structure

project_root/
├─ app/ # FastAPI backend (ML logic, API)
│ ├─ init.py
│ ├─ main.py
│ ├─ predict.py
│ └─ search_engine.py
│
├─ frontend/ # Flask web frontend (UI)
│ ├─ app.py
│ ├─ static/
│ │ └─ images/… # Category icons & logo
│ └─ templates/
│ ├─ layout.html
│ ├─ index.html
│ ├─ category.html
│ ├─ classification.html
│ └─ searching.html
│
├─ scripts/ # Data & training utilities
│ ├─ build_dataset.py
│ ├─ clean_and_reset_dataset.py
│ ├─ evaluate_classifier.py
│ ├─ remove_test_circulars.py
│ ├─ train_classifier.py
│ └─ train_semantic_classifier.py
│
├─ circulars_admin1.zip # Zipped PDFs (split archives)
├─ circulars_admin2.zip
├─ requirements.txt
└─ README.md


> Note: Trained model files (`.pkl`) are **not** committed. Train locally as needed.

---

## Prerequisites
- **Python** 3.9–3.12 (tested on 3.10)
- **pip**
- **Git**

**Linux packages** (helpful for PDF parsing):
```bash
sudo apt update
sudo apt install -y poppler-utils libgl1

SpaCy French model:

python -m spacy download fr_core_news_sm

Quick Start
1) Clone

# HTTPS
git clone https://gitlab.com/DTM-Henallux/MASI/etudiants/umm-e-farwa-leelani/finalprojectinnovation.git
cd finalprojectinnovation

# or SSH (if you set up keys)
# git clone git@gitlab.com:DTM-Henallux/MASI/etudiants/umm-e-farwa-leelani/finalprojectinnovation.git

2) Create a virtual environment

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

3) Install dependencies

pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download fr_core_news_sm

Data Preparation & Training
A) Unzip PDFs into the expected structure

Unzip the provided archives into a folder named circulars_admin/:

circulars_admin/<MainCategory>/<Subcategory>/<file>.pdf
# Example:
circulars_admin/Staff_and_Personnel/Professional_Development/example.pdf

B) Build / clean / train

From the project root:

# 1) Build dataset from PDFs (extract text, labels, circular numbers)
python scripts/build_dataset.py

# 2) (Optional) Clean/reset dataset
python scripts/clean_and_reset_dataset.py

# 3) Train classic classifier (TF-IDF + SVM)
python scripts/train_classifier.py

# 4) Train semantic classifier (sentence-transformers)
python scripts/train_semantic_classifier.py

# 5) (Optional) Evaluate performance
python scripts/evaluate_classifier.py

    After training, model artifacts will be saved locally (not in Git).
    Re-run training whenever you add new PDFs.

Run the Apps
1) Start the FastAPI backend (port 8000)

From project root:

uvicorn app.main:app --reload --port 8000
# Or, if running inside app/ directory:
# uvicorn main:app --reload --port 8000

2) Start the Flask frontend (port 5000)

cd frontend
python app.py

Open the UI at: http://localhost:5000
Usage

    Classify a PDF: use the frontend upload form; the backend returns main/subcategory + confidence.

    Semantic search: search with keywords or natural sentences (e.g., “teacher recruitment policy”).

    Browse: navigate by category/subcategory pages to view circulars.

    Duplicate detection is performed via 4-digit circular numbers parsed during dataset build.

Troubleshooting

    PDF parsing errors: ensure files are valid PDFs; on Linux install poppler-utils and libgl1.

    Model not found: you must run the training scripts before classification if the app expects local models.

    Slow first search: sentence-transformer models may load on first request.

    Windows paths: prefer running from project root; scripts use relative paths.

Roadmap

    Hybrid classification (semantic + SVM fallback)

    Docker & Docker-Compose for one-command run

    GitLab CI/CD pipeline for tests and linting

    Admin dashboard (metrics, retraining, logs)

Credits

    Author: Umm E Farwa Leelani (@Farwa1648

    )

    Supervisors: Dauvin Emmanuel, Ramonfosse Laura, Philippe Laurent

Contact

    Issues: open an Issue in this GitLab project

    Email: etu55983@henallux.be
