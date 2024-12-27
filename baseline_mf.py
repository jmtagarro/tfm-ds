import cornac
from cornac.eval_methods import RatioSplit
from cornac.datasets import movielens
from cornac.models import GlobalAvg, MostPop, MF
from cornac.metrics import Recall, NDCG
from cornac.hyperopt import Discrete
from cornac.hyperopt import GridSearch

# load the built-in MovieLens 1M and split the data based on ratio
feedback = movielens.load_feedback(variant="1M")

rs = RatioSplit(data=feedback,
                test_size=0.1,
                val_size=0.1,
                rating_threshold=4.0,
                seed=123)
mf = MF(
    max_iter=25,
    #learning_rate=0.01,
    #lambda_reg=0.02,
    use_bias=True,
    seed=123
)

rec_10 = cornac.metrics.Recall(k=10)
rec_20 = cornac.metrics.Recall(k=20)
ndcg_10 = cornac.metrics.NDCG(k=10)
ndcg_20 = cornac.metrics.NDCG(k=20)

gs_mf= GridSearch(
    model=mf,
    space=[
        Discrete(name="lambda_reg", values=[0.001, 0.002, 0.01, 0.02]),
        Discrete(name="learning_rate", values=[0.0001, 0.001, 0.01, 0.1])
    ],
    metric=ndcg_10,
    eval_method=rs,
)

models = [gs_mf]
metrics = [rec_10, rec_20, ndcg_10, ndcg_20]

# put it together in an experiment, voilà!
cornac.Experiment(eval_method=rs,
                  models=models,
                  metrics=metrics,
                  user_based=True,
                  save_dir="models"
                  ).run()

# Best parameters
print(gs_mf.best_params)