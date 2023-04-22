import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationModel:
    def __init__(self):
        self.movies_df = pd.read_csv('tmdb_5000_movies.csv')
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self._fit()

    def _fit(self):
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df['overview'].fillna(''))
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['title']).drop_duplicates()

    def get_recommendations(self, title):
        if title not in self.indices:
            return None
        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return self.movies_df['title'].iloc[movie_indices].values
