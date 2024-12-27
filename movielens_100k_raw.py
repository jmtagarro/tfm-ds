"""
Example for Disentangled Multimodal Recommendation, with feedback, textual and visual modality.
This example uses preencoded visual features from cornac dataset instead of TransformersVisionModality modality.
"""

import cornac
from cornac.data import TextModality, ImageModality
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.data import Reader
from cornac.data.text import BaseTokenizer
import numpy as np
from triton.language.extra.cuda import num_threads

# Load extracted features from posters
image_data = np.load("data/processed/posters_data.npz", allow_pickle=True)
image_features = image_data['images']
image_item_ids = image_data['ids']

# Load extracted data from subtitles
text_data = np.load("data/processed/subtitles_data.npz", allow_pickle=True)
docs = text_data['plots']
text_item_ids = text_data['ids']

# Get movie IDs present both on posters and subtitles
#common_ids = np.intersect1d(image_item_ids, text_item_ids)

# Load user-item feedback
feedback = movielens.load_feedback(variant="100K")

# only treat good feedback as positive user-item pair
#new_feedback = [f for f in feedback if f[2] >= 4.0]

text_modality = TextModality(corpus = docs, ids = text_item_ids)
image_modality = ImageModality(images=image_features, ids=image_item_ids)

ratio_split = RatioSplit(
    data=feedback,
    test_size=0.2,
    exclude_unknowns=True,
    verbose=True,
    seed=123,
    rating_threshold=4,
    item_text=text_modality,
    item_image=image_modality
)

dmrl_recommender = cornac.models.dmrl.DMRL(
    batch_size=1024,
    epochs=1,
    log_metrics=True,
    learning_rate=0.001,
    num_factors=2,
    decay_r=2,
    decay_c=0.1,
    num_neg=5,
    embedding_dim=100,
    image_dim=25088,
    dropout=0,
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

#dmrl_recommender.save(save_dir="models", save_trainset=False)