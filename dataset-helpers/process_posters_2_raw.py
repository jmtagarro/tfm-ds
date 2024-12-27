import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # For progress bar

# Directory containing the images
image_dir = "../data/ml-20m-psm/posters"  # Replace with your folder path


# Function to process a single image
def process_image(file):
    # Extract the ID from the filename (e.g., '1.jpg' -> 1)
    movie_id = int(os.path.splitext(file)[0])

    # Load the image as a Pillow object
    img_path = os.path.join(image_dir, file)
    img = Image.open(img_path).convert("RGB")  # Ensure consistent RGB format

    return movie_id, img


# Get the list of image files
image_files = [file for file in os.listdir(image_dir) if file.endswith(".jpg")]

# Use ThreadPoolExecutor to process images in parallel with a progress bar
results = []
with ThreadPoolExecutor() as executor:
    # Wrap the executor.map call with tqdm for a progress bar
    for result in tqdm(executor.map(process_image, image_files), total=len(image_files), desc="Processing Images"):
        results.append(result)

# Separate the results into IDs and images
ids, images = zip(*results)  # Unzip the results

# Convert to NumPy arrays
images_array = np.array(images, dtype=object)  # Use dtype=object for image objects
ids_array = np.array(ids)

# Save to an .npz file
np.savez("../data/processed/posters_data.npz", images=images_array, ids=ids_array)

# Print confirmation
print(f"Processed {len(ids_array)} images.")
print("Example ID:", ids_array[0])
print("Example Image Object:", images_array[0])
