import pandas as pd

# Load the links.csv file
links_file = "data/ml-20m/links.csv"
links_df = pd.read_csv(links_file)

# Create a dictionary for quick lookup of movielens_id by tmdb_id
id_map = {tmdb_id: movielens_id for movielens_id, tmdb_id in zip(links_df['movieId'], links_df['tmdbId'])}

# Load the movies_metadata.csv file
metadata_file = "data/ml-20m/movies_metadata.csv"
metadata_df = pd.read_csv(metadata_file)

# Replace 'id' (tmdbId) with movielens_id using the mapping while avoiding conflicts
metadata_df['id'] = metadata_df['id'].apply(lambda x: id_map.get(x, x) if x not in id_map.values() else x)

# Filter rows to include only those with a valid movielens_id
metadata_df = metadata_df.dropna(subset=['id'])
metadata_df['id'] = metadata_df['id'].astype(int)

# Select the required columns and reorder them with 'id' as the first column
columns_to_keep = [
    'id', 'budget', 'original_language', 'overview', 'popularity',
    'production_companies', 'production_countries', 'release_date',
    'revenue', 'spoken_languages', 'vote_average', 'vote_count'
]
metadata_df = metadata_df[columns_to_keep]

# Save the updated metadata to a new file
output_file = "metadata.csv"
metadata_df.to_csv(output_file, index=False)

print(f"Saved updated metadata to {output_file} successfully.")