import tez
import pandas as pd
from sklearn import model_selection

import torch


class MovieDataset:
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings
        
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, item):
        user = self.users[item]
        movie = self.movies[item]
        rating = self.ratings[item]
        
        return {
            "user": torch.tensor(user, dtype=torch.long),
            "movie": torch.tensor(movie, dtype=torch.long),
            "rating": torch.tensor(rating, dtype=torch.float),
            }



def train():
    df = pd.read_csv('/home/paul/Mindscope/Mindscope/Movie-Recomendation-System-FastAPI-React-/backend/ml-latest-small/ratings.csv')
    # ID, user, movie, rating
    df_train, df_valid = model_selection.train_test_split(
        df, test_size=0.1, random_state=42, stratify=df.rating.values
        )

    train_dataset = MovieDataset(
        users= df_train.userId.values, 
        movies= df_train.movieId.values, 
        ratings= df_train.rating.values 
        )
    
    valid_dataset = MovieDataset(
        users= df_valid.userId.values, 
        movies= df_valid.movieId.values, 
        ratings= df_valid.rating.values 
        )


if __name__ == '__main__':
    train()
