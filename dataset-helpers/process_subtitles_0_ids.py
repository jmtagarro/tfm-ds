import os
import pandas as pd
import shutil

# Load the links.csv file
links_file = "../data/Sublens_20M/Sublens_20M_metadata/links.csv"
df = pd.read_csv(links_file)

# Create a dictionary for quick lookup of movielens_id by imdb_id
id_map = {str(imdb_id).zfill(7): str(movielens_id) for movielens_id, imdb_id in zip(df['movieId'], df['imdbId'])}

# Define the source and destination folders
subtitles_folder = "../data/Sublens_20M/subtitles"
movielens_subs_folder = "data/Sublens_20M/movielens_subs"
os.makedirs(movielens_subs_folder, exist_ok=True)

# Iterate through all folders in the subtitles directory
for imdb_id in os.listdir(subtitles_folder):
    imdb_folder_path = os.path.join(subtitles_folder, imdb_id)

    # Ensure it's a directory and that the imdb_id exists in the mapping
    if os.path.isdir(imdb_folder_path) and imdb_id in id_map:
        movielens_id = id_map[imdb_id]

        # Look for .srt files inside the folder
        srt_files = [f for f in os.listdir(imdb_folder_path) if f.endswith('.srt')]

        if srt_files:
            srt_file = srt_files[0]  # Assuming only one .srt file per folder
            src_file_path = os.path.join(imdb_folder_path, srt_file)
            dest_file_path = os.path.join(movielens_subs_folder, f"{movielens_id}.srt")

            # Copy and rename the file
            shutil.copy(src_file_path, dest_file_path)
            print(f"Copied {src_file_path} to {dest_file_path}")

print("Subtitle files have been processed.")