
import matplotlib.pyplot as plt
import numpy as np



# Data from the table
models = ["Matrix Factorization", "GlobalAvg", "TopPop", "WBPR", "DMRL-A", "DMRL-B", "DMRL-C", "DMRL-D"]
ndcg10 = [0.0965, 0.0049, 0.1079, 0.2322, 0.2603, 0.1845, 0.1801, 0.2311]
ndcg20 = [0.1113, 0.0065, 0.1291, 0.2323, 0.2844, 0.2117, 0.2806, 0.2566]
recall10 = [0.0938, 0.0057, 0.1300, 0.1421, 0.2103, 0.1573, 0.1569, 0.1826]
recall20 = [0.1409, 0.0105, 0.1918, 0.2148, 0.3268, 0.2558, 0.2683, 0.2981]

# Index positions for the models
x = np.arange(len(models))

# Plotting NDCG
plt.figure(figsize=(10, 6))
plt.bar(x - 0.2, ndcg10, width=0.4, label="NDCG@10", hatch='/', edgecolor='black')
plt.bar(x + 0.2, ndcg20, width=0.4, label="NDCG@20", hatch='\\', edgecolor='black')
plt.xticks(x, models, rotation=45, ha="right")
plt.ylabel("NDCG")
plt.title("NDCG Comparison Across Models")
plt.legend()
plt.tight_layout()
plt.savefig("assets/ndcg_comparison.svg")

# Plotting Recall
plt.figure(figsize=(10, 6))
plt.bar(x - 0.2, recall10, width=0.4, label="Recall@10", hatch='/', edgecolor='black')
plt.bar(x + 0.2, recall20, width=0.4, label="Recall@20", hatch='\\', edgecolor='black')
plt.xticks(x, models, rotation=45, ha="right")
plt.ylabel("Recall")
plt.title("Recall Comparison Across Models")
plt.legend()
plt.tight_layout()
plt.savefig("assets/recall_comparison.svg")

# Plotting all metrics in a single chart
plt.figure(figsize=(12, 8))

plt.plot(x, ndcg10, marker='o', label="NDCG@10", linestyle='-', color='blue')
plt.plot(x, ndcg20, marker='s', label="NDCG@20", linestyle='--', color='green')
plt.plot(x, recall10, marker='^', label="Recall@10", linestyle='-', color='red')
plt.plot(x, recall20, marker='x', label="Recall@20", linestyle='--', color='purple')

plt.xticks(x, models, rotation=45, ha="right")
plt.ylabel("Score")
plt.title("Comparison of NDCG and Recall Metrics Across Models")
plt.legend()
plt.tight_layout()
plt.savefig("assets/comparison.svg")