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

class PrecomputedTextModality(TextModality):
    def __init__(self, ids, features, **kwargs):
        """
        Custom TextModality that uses precomputed embeddings.

        Parameters:
        - ids: List or array of item IDs corresponding to the embeddings.
        - features: Precomputed text embeddings as a NumPy array.
        """
        super().__init__(**kwargs)
        self.ids = np.array(ids)
        self.features = np.array(features)
        self.id_to_idx = {id_: idx for idx, id_ in enumerate(self.ids)}
        self.output_dim = self.features.shape[1]  # Set output dimension based on features
        self.preencoded = True  # Indicate that features are precomputed

    def _get_features(self, item_id):
        """
        Return the feature vector for a given item_id.
        """
        idx = self.id_to_idx.get(item_id)
        if idx is None:
            return None  # Handle missing IDs gracefully
        return self.features[idx]


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
    name="text"
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
    epochs=120,
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