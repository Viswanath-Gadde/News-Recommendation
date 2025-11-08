from flask import Flask, render_template, request
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset safely
df = pd.read_json("News_Category_Dataset_v3.json", lines=True, nrows=50000)


# Clean up missing or bad rows
df = df.dropna(subset=['headline', 'short_description', 'category']).reset_index(drop=True)

# Build TF-IDF model on the headlines
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['headline'])

app = Flask(__name__)

def content_base_rec(title, n_rows=5):
    # Clean input
    title_clean = re.sub(r"\W+", " ", title).lower().strip()

    if not title_clean:
        return []

    # Vectorize input
    title_vec = vectorizer.transform([title_clean])

    # Compute cosine similarity
    cosine_scores = cosine_similarity(title_vec, tfidf_matrix).flatten()

    # Sort and select top articles
    indices = cosine_scores.argsort()[::-1][:n_rows]

    # If similarity scores are too low, return an empty list
    if cosine_scores[indices[0]] < 0.01:
        return []

    # Return article data
    results = df.iloc[indices][['headline', 'short_description', 'category',"link"]].to_dict('records')
    return results


@app.route("/", methods=['POST', 'GET'])
def index():
    recommendations = []
    message = ""
    if request.method == "POST":
        title = request.form.get("title")
        recommendations = content_base_rec(title)
        if not recommendations:
            message = "No similar articles found. Try a broader or different headline."
    return render_template("index.html", recommendations=recommendations, message=message)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

