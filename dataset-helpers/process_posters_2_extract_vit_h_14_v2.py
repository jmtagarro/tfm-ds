import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models._api import WeightsEnum
from PIL import Image
import numpy as np

# Define device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained ViT model from torchvision
#model_weights: WeightsEnum = models.ViT_H_14_Weights.DEFAULT
model_weights: WeightsEnum = models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1

vit_model = models.vit_h_14(weights=model_weights).to(device)

# suppress the classification part
vit_model.heads = torch.nn.Identity()
vit_model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize as required by ViT
])

# Folder paths
image_folder = "../data/processed/posters"  # Replace with your posters folder
output_file = "../data/processed/posters_vit_H_14_features_v2.npz"  # Replace with the path for saving features

# Initialize lists for features and IDs
features = []
movie_ids = []

# Process each image
for file_name in os.listdir(image_folder):
    #if file_name.endswith(".jpg"):
    if file_name.endswith(".jpg"):
        print(f"Processing {file_name}")
        # Extract MovieLens ID from the file name
        movie_id = os.path.splitext(file_name)[0]
        movie_ids.append(movie_id)

        # Load and preprocess the image
        image_path = os.path.join(image_folder, file_name)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        # Extract features from the CLS token
        with torch.no_grad():
            feats = vit_model._process_input(image_tensor)
            batch_cls = vit_model.class_token.expand(feats.shape[0], -1, -1)
            feats = torch.cat([batch_cls, feats], dim=1)
            feats = vit_model.encoder(feats)
            feats = feats[:, 0][0]
            #feats = torch.flatten(feats, start_dim=1)  # Flatten the features
            features.append(feats.cpu().numpy())

# Convert lists to NumPy arrays
features_array = np.array(features)
ids_array = np.array(movie_ids)

# Save the IDs and features in an NPZ file
np.savez(output_file, ids=ids_array, features=features_array)
print(f"Features and IDs saved to {output_file}")
