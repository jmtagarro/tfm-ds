
import cornac
from cornac.data import Reader
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.models import GlobalAvg, MostPop, MF
from collections import Counter

# Load netflix dataset (small version), and binarise ratings using cornac.data.Reader
feedback = movielens.load_feedback(variant="1M")

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=feedback,
    test_size=0.1,
    rating_threshold=1.0,
    exclude_unknowns=True,
    verbose=True,
)

# Extract training and test sets
train_set = ratio_split.train_set
test_set = ratio_split.test_set

# 1. Identify Cold-Start Users (users with <= 5 interactions in the training set)
user_interaction_counts = Counter(train_set.user_ids)
cold_start_users = {user for user, count in user_interaction_counts.items() if count <= 5}

# Filter test set for cold-start users
uir_data = test_set.uir_tuple
cold_start_user_test_set = [
    (user, item, rating)
    for user, item, rating in zip(uir_data[0], uir_data[1], uir_data[2])
    if user in cold_start_users
]

# Replace the test set with cold-start users
ratio_split.test_set._uir_tuple = (
    [user for user, _, _ in cold_start_user_test_set],
    [item for _, item, _ in cold_start_user_test_set],
    [rating for _, _, rating in cold_start_user_test_set],
)

# 2. Identify Cold-Start Items (items with <= 5 interactions in the training set)
item_interaction_counts = Counter(train_set.item_ids)
cold_start_items = {item for item, count in item_interaction_counts.items() if count <= 5}

# Filter test set for cold-start items
cold_start_item_test_set = [
    (user, item, rating)
    for user, item, rating in zip(uir_data[0], uir_data[1], uir_data[2])
    if item in cold_start_items
]

# Replace the test set with cold-start items
ratio_split_item_test_set = ratio_split
ratio_split_item_test_set.test_set._uir_tuple = (
    [user for user, _, _ in cold_start_item_test_set],
    [item for _, item, _ in cold_start_item_test_set],
    [rating for _, _, rating in cold_start_item_test_set],
)


# Instantiate the most popular baseline, BPR, and WBPR models
most_pop = cornac.models.MostPop()
bpr = cornac.models.BPR(
    k=50, max_iter=200, learning_rate=0.001, lambda_reg=0.001, verbose=True
)
wbpr = cornac.models.WBPR(
    k=50, max_iter=200, learning_rate=0.001, lambda_reg=0.001, verbose=True
)
global_avg = GlobalAvg()
mf = MF(max_iter=25, learning_rate=0.0001, lambda_reg=0.002, use_bias=True, seed=123)

# Use AUC and Recall@20 for evaluation
rec_10 = cornac.metrics.Recall(k=10)
ndcg_10 = cornac.metrics.NDCG(k=10)
rec_20 = cornac.metrics.Recall(k=20)
ndcg_20 = cornac.metrics.NDCG(k=20)

# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=ratio_split,
    models=[most_pop, bpr, wbpr, global_avg, mf],
    metrics=[ndcg_10, ndcg_20, rec_10, rec_20],
    user_based=True,
    save_dir="models"
).run()