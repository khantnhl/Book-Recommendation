import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import Birch
import pickle


def train_tfidf(df, column_name, ngram_range=(1, 2), min_df=0.01, max_df=0.99):
    
    """Train TF-IDF vectorizer on a specified column."""
    tfidf = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)
    tfidf_matrix = tfidf.fit_transform(df[column_name])
    return tfidf, tfidf_matrix

def train_scaler(tfidf_matrix):
    """Train a MaxAbsScaler on the TF-IDF matrix."""
    scaler = MaxAbsScaler()
    scaled_matrix = scaler.fit_transform(tfidf_matrix)
    return scaler, scaled_matrix

def train_svd(scaled_matrix, n_components=3):
    """Train Truncated SVD on the scaled matrix."""
    svd = TruncatedSVD(n_components=n_components, algorithm='arpack')
    reduced_matrix = svd.fit_transform(scaled_matrix)
    return svd, reduced_matrix

def train_birch(reduced_matrix, threshold=0.01, n_clusters=25):
    """Train a BIRCH clustering model."""
    birch = Birch(threshold=threshold, n_clusters=n_clusters)
    birch.fit(reduced_matrix)
    return birch

def save_model(model, filepath):
    """Save the model to a specified filepath."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

# Main Training Pipeline

def main():
    # Load the dataset
    df = pd.read_csv('data/raw/books_data.csv')

    # Merge text fields for TF-IDF training
    df['merged'] = df['title'] + ' ' + df['subjects'] + ' ' + df['synopsis']

    # Train TF-IDF Vectorizer
    print("Training TF-IDF...")
    tfidf, tfidf_matrix = train_tfidf(df, 'merged')
    save_model(tfidf, 'models/tfidf_vectorizer.pkl')

    # Train MaxAbsScaler
    print("Training MaxAbsScaler...")
    scaler, scaled_matrix = train_scaler(tfidf_matrix)
    save_model(scaler, 'models/maxabs_scaler.pkl')

    # Train Truncated SVD
    print("Training TruncatedSVD...")
    svd, reduced_matrix = train_svd(scaled_matrix, n_components=3)
    save_model(svd, 'models/svd_model.pkl')

    # Train BIRCH
    print("Training BIRCH...")
    birch = train_birch(reduced_matrix, threshold=0.01, n_clusters=25)
    save_model(birch, 'models/birch_model.pkl')

    # Save reduced data for later use
    reduced_df = pd.DataFrame(reduced_matrix, index=df.index)
    reduced_df.to_pickle('data/processed/reduced_data.pkl')

    print("All models trained and saved successfully!")

# Entry Point

if __name__ == "__main__":
    main()
