import numpy as np
from sentence_transformers import SentenceTransformer


def load_subtitles(npz_path):
    """
    Load subtitle IDs and plots from the specified .npz file.
    """
    data = np.load(npz_path, allow_pickle=True)
    ids = data['ids']
    plots = data['plots']
    return ids, plots


def extract_embeddings(plots, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"):
    """
    Extract embeddings for the given plots using the specified SentenceTransformer model.
    Automatically uses CUDA if available.
    Returns a NumPy array of embeddings.
    """
    # Load the SentenceTransformer model with CUDA if available
    model = SentenceTransformer(model_name)
    device = "cuda" if model.device.type == "cuda" else "cpu"
    print(f"Using device: {device}")

    # Compute embeddings for all plots
    embeddings = model.encode(
        plots,
        convert_to_numpy=True,
        show_progress_bar=True,
        device=device
    )

    return embeddings


def save_embeddings(output_path, ids, embeddings):
    """
    Save embeddings and IDs to a .npz file.
    """
    np.savez(output_path, ids=ids, features=embeddings)
    print(f"Embeddings saved to {output_path}")


if __name__ == "__main__":
    # Path to the .npz file containing subtitle data
    input_npz_path = "../data/processed/subtitles_data.npz"  # Replace with your .npz file path
    output_npz_path = "../data/processed/subtitles_st_p_miniLM_L6_v2_features.npz"  # Path to save embeddings

    # Load subtitles
    ids, plots = load_subtitles(input_npz_path)

    # Extract embeddings
    embeddings = extract_embeddings(plots, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

    # Save embeddings
    save_embeddings(output_npz_path, ids, embeddings)
