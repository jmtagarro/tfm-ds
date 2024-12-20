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

# Load extracted features from posters
image_data = np.load("data/processed/posters_vit_L_32_features.npz")
image_features = image_data['features']
image_item_ids = image_data['ids']

# Load extracted data from subtitles
text_data = np.load("data/processed/subtitles_data.npz", allow_pickle=True)
docs = text_data['plots']
text_item_ids = text_data['ids']

# Get movie IDs present both on posters and subtitles
common_ids = np.intersect1d(image_item_ids, text_item_ids)

# Load user-item feedback
feedback = movielens.load_feedback(variant="1M", reader=Reader(item_set = common_ids))

# only treat good feedback as positive user-item pair
#new_feedback = [f for f in feedback if f[2] >= 4.0]

text_modality = TextModality(corpus = docs, ids = text_item_ids,
                             tokenizer = BaseTokenizer(sep=' ', stop_words = 'english'),
                             max_vocab = 5000, max_doc_freq = 0.5)

image_modality = ImageModality(features=image_features, ids=image_item_ids)

ratio_split = RatioSplit(
    data=feedback,
    test_size=0.25,
    exclude_unknowns=True,
    verbose=True,
    seed=123,
    rating_threshold=4,
    item_text=text_modality,
    item_image=image_modality,
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
    image_dim=1024,
    dropout=0,
)


# Use Recall@300 for evaluations
rec_10 = cornac.metrics.Recall(k=10)
rec_20 = cornac.metrics.Recall(k=20)
rec_50 = cornac.metrics.Recall(k=50)
rec_100 = cornac.metrics.Recall(k=100)
ndcg_10 = cornac.metrics.NDCG(k=10)
ndcg_20 = cornac.metrics.NDCG(k=20)
ndcg_50 = cornac.metrics.NDCG(k=50)
ndcg_100 = cornac.metrics.NDCG(k=100)

# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=ratio_split,
    models=[dmrl_recommender],
    metrics=[rec_10, rec_20, rec_50, rec_100, ndcg_10, ndcg_20, ndcg_50, ndcg_100]
).run()