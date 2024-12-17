import pandas as pd

# Load the links.csv file
links_file = "data/ml-20m/links.csv"
links_df = pd.read_csv(links_file)

# Create a dictionary for quick lookup of movielens_id by tmdb_id
id_map = {tmdb_id: movielens_id for movielens_id, tmdb_id in zip(links_df['movieId'], links_df['tmdbId'])}

# Load the credits.csv file
credits_file = "data/ml-20m/credits.csv"
credits_df = pd.read_csv(credits_file)

# Replace 'id' values using the mapping while avoiding conflicts
credits_df['id'] = credits_df['id'].map(lambda x: id_map.get(x, x))

# Reorder the columns to id, cast, crew
credits_df = credits_df[['id', 'cast', 'crew']]

# Save the updated CSV to a new file
new_credits_file = "data/ml-20m/credits2.csv"
credits_df.to_csv(new_credits_file, index=False)

print(f"Saved updated data to {new_credits_file} successfully.")