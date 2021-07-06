from rdkit import Chem
import deepchem as dc
import numpy as np


data = dc.data.datasets.NumpyDataset.from_json("dataset_out")
#data.from_json("dataset_out")
# Splits dataset into train/validation/test
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=data)
task_count = len(data.y[0])


metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression")
]

params_dict = {
    'n_tasks': [task_count],
    'n_embedding': [[40], [100], [1000]],
    'dropouts': [0.1, 0.2, 0.5],
    'learning_rate': [0.001, 0.0001]
}

optimizer = dc.hyper.GridHyperparamOpt(dc.models.DTNNModel)
fit_transformers = [dc.trans.CoulombFitTransformer(data)]
best_model, best_hyperparams, all_results =  optimizer.hyperparam_search(params_dict, train_dataset, valid_dataset,
                                                                         metric, fit_transformers)

model = dc.models.DTNNModel(
    n_tasks=task_count,
    n_embedding=10,
    mode="regression",
    dropout=0.1,
    learning_rate=0.1
)

print(all_results)

model.fit(train_dataset)
# How well the model fit's the training subset of our data
train_scores = model.evaluate(train_dataset, metric)
# Validation of the model over several training iterations.
valid_score = model.evaluate(valid_dataset, metric)
# How well the model generalizes the rest of the data
test_score = model.evaluate(test_dataset, metric)
print("Training Scores: ")
print(train_scores)
print("Validity Scores: ")
print(valid_score)
print("Test Scores: ")
print(test_score)


