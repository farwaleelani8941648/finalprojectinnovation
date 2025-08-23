import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    """
    Semantic search engine for circulars using sentence embeddings and FAISS.
    Provides cosine similarity search over categorized document content.
    """
    def __init__(self, dataset_path="dataset/circulars_dataset.csv", top_k=5, score_threshold=0.4):
        print("Initializing Semantic Search Engine with cosine similarity...")

        self.top_k = top_k
        self.score_threshold = score_threshold

        # Load dataset
        self.df = pd.read_csv(dataset_path)

        # Concatenate category, subcategory, and content for richer embedding
        self.df["content_plus"] = (
            self.df["main_category"].fillna('') + " " +
            self.df["subcategory"].fillna('') + " " +
            self.df["content"]
        )

        # Load pre-trained sentence transformer model
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        # Generate normalized embeddings for all documents
        print("Generating embeddings...")
        embeddings = self.model.encode(self.df["content_plus"].tolist(), show_progress_bar=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / norms

        # Create FAISS index for efficient cosine similarity search
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype("float32"))

        print(f"FAISS index created with {len(self.embeddings)} circulars.")

    def search(self, query, top_k=None):
        """
        Search for similar circulars based on a text query.
        Returns the top_k most similar entries from the dataset.
        """
        if top_k is None:
            top_k = self.top_k

        # Encode and normalize the query
        query_vec = self.model.encode([query])[0].astype("float32")
        query_vec /= np.linalg.norm(query_vec)

        # Perform FAISS search
        scores, indices = self.index.search(np.array([query_vec]), top_k)

        results = []
        print("\n=== DEBUG: FAISS Raw Matches ===")
        for idx, score in zip(indices[0], scores[0]):
            print(f"{self.df.iloc[idx]['filename']} - Score: {score:.3f}")
            # To filter by score threshold, uncomment the next line
            # if score >= self.score_threshold:
            row = self.df.iloc[idx].to_dict()
            row["score"] = float(score)
            results.append(row)

        if not results:
            print("No results passed the score threshold.")

        return results
