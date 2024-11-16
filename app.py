from flask import Flask
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define the RecommendationModel class
class RecommendationModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.book_data = None
        self.book_vectors = None

    def train(self, book_data):
        self.book_data = book_data
        self.book_vectors = self.vectorizer.fit_transform(book_data["description"])

    def recommend(self, query, top_n=5):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.book_vectors).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]
        return self.book_data.iloc[top_indices]["title"].tolist()

# Load the model
try:
    recommender = joblib.load("data/recommendation_model.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading the model: {e}")
    recommender = None

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the Recommendation System API!"

if __name__ == "__main__":
    app.run(debug=True)
