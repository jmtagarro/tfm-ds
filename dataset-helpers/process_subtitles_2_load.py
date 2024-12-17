import os
import numpy as np

# Directory containing subtitle files
subtitle_dir = "../data/processed/subtitles"  # Replace with your folder path

# Lists to store IDs and content
ids = []
plots = []

# Loop through all files in the folder
for file in os.listdir(subtitle_dir):
    if file.endswith(".txt"):
        # Extract the ID from the filename (e.g., '123.txt' -> 123)
        movie_id = int(os.path.splitext(file)[0])
        ids.append(movie_id)

        # Read the content of the subtitle file
        file_path = os.path.join(subtitle_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Append content to plots
        plots.append(content)

# Convert to NumPy arrays for later processing
ids_array = np.array(ids)
plots_array = np.array(plots, dtype=object)  # Use dtype=object for text data

# Save arrays for later use
np.savez("../data/processed/subtitles_data.npz", ids=ids_array, plots=plots_array)

# Print confirmation
print(f"Processed {len(ids)} subtitle files.")
print("Example ID:", ids_array[0])
print("Example Content:", plots_array[0][:200])  # Print the first 200 characters
