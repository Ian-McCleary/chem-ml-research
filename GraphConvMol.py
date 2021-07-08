from rdkit import Chem
import deepchem as dc
import numpy as np
import pandas as pd
import tempfile


# update task count as list ["task1", "task2"..]
loader = dc.data.CSVLoader(["task1"], feature_field="smiles", id_field="ids", featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False))
data = loader.create_dataset("dataset_out.csv")

# Splits dataset into train/validation/test
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=data)
task_count = len(train_dataset.y[0])

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

