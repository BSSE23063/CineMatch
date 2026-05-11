import pandas as pd



# load data

movies = pd.read_csv('ml-32m/movies.csv')
ratings = pd.read_csv('ml-32m/ratings.csv')
tags = pd.read_csv('ml-32m/tags.csv')


# merging data

# merge movies and ratings
df = pd.merge(movies, ratings, on='movieId', how='inner')

print(df.head())

df.to_csv('processed_ratings.csv', index=False)
print("Done! 'processed_ratings.csv' created.")