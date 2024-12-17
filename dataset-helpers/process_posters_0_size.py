import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def resize_and_pad_image(input_path, output_path, size=(224, 224)):
    """
    Resizes an image to fit within the target size while preserving the aspect ratio.
    Adds black bars (padding) to maintain the desired dimensions.
    Uses bicubic interpolation for resizing.
    """
    try:
        # Open the image
        with Image.open(input_path) as img:
            # Convert to RGB (ensure compatibility with JPEG)
            img = img.convert("RGB")

            # Resize image while preserving aspect ratio using bicubic interpolation
            img.thumbnail((size[0], size[1]), Image.BICUBIC)

            # Create a new black image with target size
            new_img = Image.new("RGB", size, (0, 0, 0))

            # Center the resized image on the black canvas
            paste_x = (size[0] - img.size[0]) // 2
            paste_y = (size[1] - img.size[1]) // 2
            new_img.paste(img, (paste_x, paste_y))

            # Save the resulting image
            new_img.save(output_path, "JPEG")
            print(f"Processed: {os.path.basename(input_path)}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def process_images_in_parallel(input_folder, output_folder, target_size=(224, 224), max_workers=16):
    """
    Processes all JPEG images in the input folder using multiple threads,
    resizes them to the target size with black bars, and saves them to the output folder.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all JPEG files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg'))]

    # Prepare a list of input-output paths
    tasks = []
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)
        tasks.append((input_path, output_path))

    # Process images in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(resize_and_pad_image, task[0], task[1], target_size) for task in tasks]

        # Wait for all threads to complete
        for future in futures:
            future.result()


# Define input and output folders
input_folder = "../data/ml-20m-psm/posters"  # Replace with the path to your input folder
output_folder = "../data/processed/posters"  # Replace with the path to your output folder

# Run the script
if __name__ == "__main__":
    process_images_in_parallel(input_folder, output_folder, target_size=(224, 224), max_workers=8)
