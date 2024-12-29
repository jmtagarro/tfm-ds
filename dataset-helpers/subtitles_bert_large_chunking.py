import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch


def load_subtitles(npz_path):
    """
    Load subtitle IDs and plots from the specified .npz file.
    """
    data = np.load(npz_path, allow_pickle=True)
    ids = data['ids']
    plots = data['plots']
    return ids, plots


def encode_long_text(text, model_name="bert-base-uncased", chunk_size=512, device="cuda"):
    """
    Encode long text by splitting it into chunks and aggregating the embeddings.
    Aggregation is done using mean pooling across all chunks.
    Automatically uses CUDA if available.
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Split text into chunks
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=chunk_size,
        return_overflowing_tokens=True
    )

    # Process each chunk
    chunk_embeddings = []
    with torch.no_grad():
        for i in range(len(inputs['input_ids'])):
            chunk = {key: val[i].unsqueeze(0).to(device) for key, val in inputs.items() if key in ['input_ids', 'attention_mask']}
            outputs = model(**chunk)
            chunk_embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling over tokens
            chunk_embeddings.append(chunk_embedding)

    # Aggregate chunk embeddings (mean pooling across chunks)
    aggregated_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0)
    return aggregated_embedding.squeeze(0).cpu().numpy()


def save_embeddings(output_path, ids, embeddings):
    """
    Save embeddings and IDs to a .npz file.
    """
    np.savez(output_path, ids=ids, features=embeddings)
    print(f"Embeddings saved to {output_path}")


if __name__ == "__main__":
    # Path to the .npz file containing subtitle data
    input_npz_path = "../data/processed/subtitles_data.npz"  # Replace with your .npz file path
    output_npz_path = "../data/processed/subtitles_bert_large_chunking_features.npz"  # Path to save embeddings

    # Load subtitles
    ids, plots = load_subtitles(input_npz_path)

    # Define model and chunk size
    model_name = "bert-base-uncased"  # You can replace with any BERT model
    chunk_size = 512

    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Generate embeddings for all subtitles
    embeddings = []
    for i, text in enumerate(plots):
        print(f"Processing subtitle {i+1}/{len(plots)}...")
        embedding = encode_long_text(text, model_name=model_name, chunk_size=chunk_size, device=device)
        embeddings.append(embedding)

    embeddings = np.array(embeddings)

    # Save embeddings
    save_embeddings(output_npz_path, ids, embeddings)
