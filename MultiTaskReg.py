from rdkit import Chem
import deepchem as dc
import numpy as np

data = dc.data.datasets.NumpyDataset.from_json("dataset_out")
# Splits dataset into train/validation/test
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=data)
task_count = len(data.y[0])