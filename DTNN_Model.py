from rdkit import Chem
import deepchem as dc
import numpy as np


data = dc.data.datasets.NumpyDataset.from_json("dataset_out")
#data.from_json("dataset_out")
# Splits dataset into train/validation/test
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=data)
task_count = len(data.y[0])


metrics = [
    dc.metrics.Metric(dc.metrics.rms_score),
    dc.metrics.Metric(dc.metrics.r2_score)
    ]
metric = dc.metrics.Metric(dc.metrics.rms_score)

# parameter optimization
params_dict = {
    'n_tasks': [task_count],
    'n_embedding': [5, 10, 50, 100],
    'dropouts': [0.1, 0.2, 0.5, 0.9],
    'learning_rate': [0.001, 0.0001, 0.00001, 0.000001]
}

optimizer = dc.hyper.GridHyperparamOpt(dc.models.DTNNModel)
transformers = [dc.trans.NormalizationTransformer(transform_y=True, dataset=data)]
best_model, best_hyperparams, all_results =  optimizer.hyperparam_search(params_dict, train_dataset, valid_dataset,
                                                                         metric, transformers)                                                                  
print(all_results)
print("\n")
print(best_model)
print("\n")
print(best_hyperparams)

# Single evaluation model
'''
model = dc.models.DTNNModel(
    n_tasks=task_count,
    n_embedding=1000,
    mode="regression",
    dropout=0.5,
    learning_rate=0.0001
)


model.fit(train_dataset)
# How well the model fit's the training subset of our data
train_scores = model.evaluate(train_dataset, metrics)
# Validation of the model over several training iterations.
valid_score = model.evaluate(valid_dataset, metrics)
# How well the model generalizes the rest of the data
test_score = model.evaluate(test_dataset, metrics)
print("Training Scores: ")
print(train_scores)
print("Validity Scores: ")
print(valid_score)
print("Test Scores: ")
print(test_score)
'''