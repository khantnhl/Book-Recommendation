import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(input_vector, cluster_matrix):
    """
    Calculate cosine similarity between the input vector and items in the cluster.
    Args:
        input_vector (array-like): A single input vector (1D or 2D array).
        cluster_matrix (array-like): Matrix of vectors in the cluster (2D array).
    Returns:
        ndarray: A 1D array of similarity scores.
    """
    return cosine_similarity(input_vector, cluster_matrix).flatten()

def append_similarity_to_subset(input_vector, subset_df, cluster_matrix):
    """
    Append similarity scores to a subset DataFrame.
    Args:
        input_vector (array-like): A single input vector (1D or 2D array).
        subset_df (pd.DataFrame): The subset of data being compared.
        cluster_matrix (array-like): Matrix of vectors in the cluster (2D array).
    Returns:
        pd.DataFrame: Subset DataFrame with an added 'similarity' column.
    """
    similarity_scores = calculate_similarity(input_vector, cluster_matrix)
    subset_df = subset_df.copy()
    subset_df['similarity'] = similarity_scores
    return subset_df

def get_top_k_similar(subset_df, k=5):
    """
    Retrieve the top k most similar items from the subset DataFrame.
    Args:
        subset_df (pd.DataFrame): Subset DataFrame with a 'similarity' column.
        k (int): Number of top items to return.
    Returns:
        pd.DataFrame: DataFrame with the top k similar items, sorted by similarity.
    """
    return subset_df.sort_values(by='similarity', ascending=False).head(k)
