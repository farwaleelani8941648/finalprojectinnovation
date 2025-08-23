from flask import Flask, render_template, request
import os
import requests
import pandas as pd

app = Flask(__name__)

# === Configuration ===
DATASET_PATH = "../dataset/circulars_dataset.csv"
PDF_ROOT = "../circulars_admin"
CLASSIFICATION_BACKEND = "http://localhost:8000/classify/"
SEARCH_BACKEND = "http://localhost:8000/search/"

# === Load Dataset ===
df = pd.read_csv(DATASET_PATH)

def build_navigation():
    nav = {}
    for main in df["main_category"].dropna().unique():
        subs = df[df["main_category"] == main]["subcategory"].dropna().unique().tolist()
        nav[main] = subs
    return nav

nav = build_navigation()

def get_all_circulars():
    all_circulars = {}
    for main in df["main_category"].dropna().unique():
        all_circulars[main] = {}
        for sub in df[df["main_category"] == main]["subcategory"].dropna().unique():
            filtered = df[(df["main_category"] == main) & (df["subcategory"] == sub)]
            circulars = [
                {
                    "filename": row["filename"],
                    "path": os.path.join(PDF_ROOT, main, sub, row["filename"]),
                    "summary": (row["content"][:300] + "...") if len(row["content"]) > 300 else row["content"]
                }
                for _, row in filtered.iterrows()
            ]
            all_circulars[main][sub] = circulars
    return all_circulars

@app.route("/")
def index():
    return render_template(
        "index.html",
        navigation=nav,
        all_circulars=get_all_circulars(),
        results=[],
        result=None,
        message=None
    )

@app.route("/classification", methods=["GET", "POST"])
def classification():
    result, results, message = None, [], None

    if request.method == "POST" and "pdf" in request.files:
        file = request.files["pdf"]
        try:
            response = requests.post(
                CLASSIFICATION_BACKEND,
                files={"file": (file.filename, file.stream, file.mimetype)}
            )
            if response.ok:
                result = response.json()
                results = result.get("matches", [])
                circ_num = result.get("circular_number")

                if result.get("existing"):
                    message = f"✅ Existing circular recognized (#{circ_num})."
                elif circ_num:
                    message = f"🟡 Circular number {circ_num} found, but not matched. Treated as new."
                else:
                    message = "🧠 New circular. AI predicted category below."
            else:
                message = "❌ Error: Unable to classify the circular."
        except Exception as e:
            message = f"❌ Exception during classification: {e}"

    return render_template(
        "classification.html",
        navigation=nav,
        results=results,
        result=result,
        message=message,
        horizontal_cards=True
    )

@app.route("/search", methods=["GET", "POST"])
def search():
    # Handles both GET (from navbar or direct link) and POST (from search form)
    results, result, message = [], None, None

    if request.method == "POST":
        query = request.form.get("query")
        if not query:
            message = "⚠️ Veuillez saisir un terme de recherche."
        else:
            try:
                response = requests.post(SEARCH_BACKEND, data={"query": query})
                if response.ok:
                    results = response.json()
                    message = f"🔍 Résultats pour : '{query}'"
                else:
                    message = "❌ Erreur du moteur de recherche."
            except Exception as e:
                message = f"❌ Exception pendant la recherche : {e}"

    # If GET, just show empty search bar (no results, no message)
    return render_template(
        "searching.html",
        navigation=nav,
        results=results,
        result=result,
        message=message,
        horizontal_cards=True
    )

@app.route("/category/<main>/<sub>")
def view_category(main, sub):
    filtered = df[(df["main_category"] == main) & (df["subcategory"] == sub)]
    circulars = [
        {
            "filename": row["filename"],
            "path": os.path.join(PDF_ROOT, main, sub, row["filename"]),
            "summary": (row["content"][:300] + "...") if len(row["content"]) > 300 else row["content"]
        }
        for _, row in filtered.iterrows()
    ]
    return render_template(
        "category.html",
        navigation=nav,
        main=main,
        sub=sub,
        circulars=circulars,
        horizontal_cards=True
    )

if __name__ == "__main__":
    app.run(debug=True)
