from rdkit import Chem
import deepchem as dc
import numpy as np

data = dc.data.datasets.NumpyDataset.from_json("dataset_out")
transformer = dc.trans.NormalizationTransformer(transform_y=True, dataset=data)
dataset = transformer.transform(data)

# Splits dataset into train/validation/test
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=dataset)
task_count = len(dataset.y[0])
n_features = len(dataset.X[0])



params_dict = {
    'n_tasks': [task_count],
    'n_features': [n_features],
    'layer_sizes': [[500], [1000], [1000, 1000]],
    'dropouts': [0.2, 0.5],
    'learning_rate': [0.001, 0.0001]
}
print(data.y)
print(data.y[0])
print(data.y[1])

optimizer = dc.hyper.GridHyperparamOpt(dc.models.MultitaskClassifier)
metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
        params_dict, train_dataset, valid_dataset, metric, transformer)
print(all_results)