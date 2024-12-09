import pickle
import numpy as np
from sklearn.decomposition import PCA
from flask import Flask, request, jsonify, render_template

# Load precomputed image embeddings
with open("image_embeddings.pickle", "rb") as f:
    data = pickle.load(f)
    image_paths = data["paths"]
    embeddings = np.array(data["embeddings"])

app = Flask(__name__)

# Helper functions
def cosine_similarity(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a, b.T)

def search_by_text(query_embedding, k=5):
    sims = cosine_similarity(query_embedding.reshape(1, -1), embeddings)
    top_k = np.argsort(-sims[0])[:k]
    return [{"path": image_paths[i], "score": sims[0, i]} for i in top_k]

def search_by_image(image_embedding, k=5):
    return search_by_text(image_embedding, k)

def search_combined(query_embedding, image_embedding, alpha, k=5):
    combined_embedding = alpha * query_embedding + (1 - alpha) * image_embedding
    return search_by_text(combined_embedding, k)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    query_type = request.form.get("type")
    top_k = int(request.form.get("top_k", 5))
    use_pca = request.form.get("use_pca") == "true"
    pca_components = int(request.form.get("pca_components", embeddings.shape[1]))

    # Optionally apply PCA
    if use_pca:
        pca = PCA(n_components=pca_components)
        reduced_embeddings = pca.fit_transform(embeddings)
    else:
        reduced_embeddings = embeddings

    # Search based on query type
    if query_type == "text":
        text_embedding = np.random.rand(reduced_embeddings.shape[1])  # Replace with actual model inference
        results = search_by_text(text_embedding, top_k)
    elif query_type == "image":
        image_embedding = np.random.rand(reduced_embeddings.shape[1])  # Replace with actual model inference
        results = search_by_image(image_embedding, top_k)
    elif query_type == "combined":
        alpha = float(request.form.get("alpha", 0.5))
        text_embedding = np.random.rand(reduced_embeddings.shape[1])  # Replace with actual model inference
        image_embedding = np.random.rand(reduced_embeddings.shape[1])  # Replace with actual model inference
        results = search_combined(text_embedding, image_embedding, alpha, top_k)
    else:
        results = []

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
