"""
Example for Disentangled Multimodal Recommendation, with feedback, textual and visual modality.
This example uses preencoded visual features from cornac dataset instead of TransformersVisionModality modality.
"""

import cornac
from cornac.data import TextModality, ImageModality
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.data import Reader
from cornac.data.text import TextModality
from cornac.data.modality import FeatureModality
from cornac import datasets
import numpy as np
import torch

class PrecomputedTextModality(TextModality):
    def __init__(self, ids, features=None, device="cpu", **kwargs):
        """
        Custom TextModality that uses precomputed embeddings.

        Parameters:
        - ids: List or array of item IDs corresponding to the embeddings.
        - features: Precomputed text embeddings as a NumPy array or PyTorch tensor.
        - device: The device ('cpu' or 'cuda') to which tensors will be moved.
        """
        super().__init__(ids=ids, **kwargs)

        # Initialize features, allowing None during parent initialization
        if features is not None:
            self._initialize_features(features, device)
        else:
            self.features_tensor = None
            self.features_numpy = None

        self.ids = np.array(ids)
        self.id_to_idx = {id_: idx for idx, id_ in enumerate(self.ids)}
        self.preencoded = True  # Indicate that features are precomputed
        self.output_dim = self.features_tensor.shape[1] if self.features_tensor is not None else 0
        self.device = device  # Store the device
        self._return_numpy = False  # Default to tensor mode

    def _initialize_features(self, features, device):
        """
        Initialize features as both a PyTorch tensor and a NumPy array.
        """
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        elif not isinstance(features, torch.Tensor):
            raise TypeError("Features must be a NumPy array or PyTorch tensor.")

        self.features_tensor = features.to(device)
        self.features_numpy = self.features_tensor.cpu().numpy()

    @property
    def features(self):
        """
        Dynamically return the appropriate format of features based on the context.
        """
        if self.features_numpy is None or self.features_tensor is None:
            return None
        return self.features_numpy if self._return_numpy else self.features_tensor

    @features.setter
    def features(self, value):
        """
        Dynamically set features while maintaining consistency across formats.
        """
        if value is None:
            self.features_tensor = None
            self.features_numpy = None
        else:
            self._initialize_features(value, self.device)

    def batch_feature(self, batch_ids):
        """
        Return a matrix (batch of feature vectors) corresponding to provided batch_ids as a PyTorch tensor.
        """
        if self.features_tensor is None:
            raise ValueError("Features are not initialized.")
        idxs = [self.id_to_idx[id_] for id_ in batch_ids if id_ in self.id_to_idx]
        return self.features_tensor[idxs]  # Always return tensors for DMRL compatibility

    def build(self, id_map, uid_map=None, iid_map=None, dok_matrix=None):
        """
        Override the build method to ensure proper initialization.
        This method ensures the modality recognizes pre-encoded features.

        Parameters:
        - id_map: Mapping of item IDs to internal indices.
        - uid_map: Optional mapping of user IDs.
        - iid_map: Optional mapping of item IDs.
        - dok_matrix: Optional document-term matrix.
        """
        if self.features_numpy is None:
            raise ValueError("Features are not initialized.")
        self._return_numpy = True  # Use NumPy format for RatioSplit
        super().build(id_map=id_map)
        self._return_numpy = False  # Reset to tensor mode for training/evaluation




# Load extracted features from posters
image_data = np.load("data/processed/posters_vit_H_14_features_v2.npz")
image_features = image_data['features']
image_item_ids = image_data['ids']

# Load extracted data from subtitles
text_data = np.load("data/processed/subtitles_bert_large_chunking_features.npz", allow_pickle=True)
text_features = text_data['features']
text_item_ids = text_data['ids']

text_modality = PrecomputedTextModality(
    ids=text_item_ids,
    features=text_features,
    device="cuda"
)

# Get movie IDs present both on posters and subtitles
common_ids = np.intersect1d(image_item_ids, text_item_ids)

# Load user-item feedback
feedback = movielens.load_feedback(variant="100K", reader=Reader(item_set = common_ids))

# only treat good feedback as positive user-item pair
#new_feedback = [f for f in feedback if f[2] >= 4.0]



image_modality = ImageModality(features=image_features, ids=image_item_ids)

ratio_split = RatioSplit(
    data=feedback,
    test_size=0.25,
    exclude_unknowns=True,
    verbose=True,
    seed=123,
    rating_threshold=4,
    item_text=text_modality,
    item_image=image_modality
)

dmrl_recommender = cornac.models.dmrl.DMRL(
    batch_size=1024,
    epochs=60,
    log_metrics=True,
    learning_rate=0.001,
    num_factors=2,
    decay_r=2,
    decay_c=0.1,
    num_neg=5,
    embedding_dim=100,
    image_dim=1280,
    dropout=0,
    bert_text_dim=768
)


# Use Recall@300 for evaluations
rec_10 = cornac.metrics.Recall(k=10)
rec_20 = cornac.metrics.Recall(k=20)
ndcg_10 = cornac.metrics.NDCG(k=10)
ndcg_20 = cornac.metrics.NDCG(k=20)

# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=ratio_split,
    models=[dmrl_recommender],
    metrics=[ndcg_10, ndcg_20, rec_10, rec_20]
).run()