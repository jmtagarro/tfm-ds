import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

def extract_vgg16_features(image_folder, output_path):
    """
    Extracts features from images using a pretrained VGG16 model.
    Assumes images are named in the format id.jpg and have a size of 224x224.
    Saves the features as a numpy array to the specified output path.

    Parameters:
    - image_folder: Path to the folder containing the images.
    - output_path: Path to save the resulting features as a numpy file.
    """
    # Define device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained VGG16 model and remove the final classification layer
    model = models.vgg16(pretrained=True)
    model = nn.Sequential(*list(model.features.children()))
    model = model.to(device)
    model.eval()

    # Define image transformation: Normalize for VGG16 and ensure 224x224 size
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL image to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # VGG16 normalization
    ])

    # Prepare lists for IDs and extracted features
    image_ids = []
    features_list = []

    # Iterate over images in the folder
    for file_name in os.listdir(image_folder):
        if file_name.lower().endswith(".jpg"):
            try:
                # Extract ID from file name
                image_id = os.path.splitext(file_name)[0]
                image_ids.append(image_id)

                # Load and transform image
                image_path = os.path.join(image_folder, file_name)
                image = Image.open(image_path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

                # Extract features
                with torch.no_grad():
                    feature = model(image_tensor)
                    feature = torch.flatten(feature, start_dim=1)  # Flatten the features
                    features_list.append(feature.cpu().numpy())

                print(f"Processed: {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    # Combine features into a single NumPy array
    features_array = np.vstack(features_list)
    ids_array = np.array(image_ids)

    # Save features and IDs to a NumPy file
    np.savez(output_path, ids=ids_array, features=features_array)
    print(f"Features saved to {output_path}")

# Define input folder and output file path
image_folder = "../data/processed/posters"   # Replace with the path to your image folder
output_file = "../data/processed/posters_vgg16_features.npz"  # Replace with the desired output file path

# Run the feature extraction
if __name__ == "__main__":
    extract_vgg16_features(image_folder, output_file)
