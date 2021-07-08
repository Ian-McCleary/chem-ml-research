from rdkit import Chem
import deepchem as dc
import numpy as np


data = dc.data.datasets.NumpyDataset.from_json("dataset_out")

# Splits dataset into train/validation/test
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=data)
task_count = len(data.y[0])

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

# supports width of channels from convolutional layers and dropout for each layer.
model = dc.model.GraphConvModel(
  n_tasks=task_count,
  number_atom_features=75,
  mode="regression"
)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric])
valid_scores = model.evaluate(valid_dataset, [metric])

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

