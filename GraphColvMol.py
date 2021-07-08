from rdkit import Chem
import deepchem as dc
import numpy as np

'''
data = dc.data.datasets.NumpyDataset.from_json("dataset_out")

# Splits dataset into train/validation/test
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=data)
task_count = len(data.y[0])

metric = dc.metrics.Metric(dc.metrics.rms_score)

model = dc.model.GraphConvModel(

)
'''
import matplotlib.pyplot as plt


np.random.seed(123)
import tensorflow as tf

tf.random.set_seed(123)
from deepchem.molnet import load_delaney

# Load Delaney dataset
delaney_tasks, delaney_datasets, transformers = load_delaney(
    featurizer='GraphConv', split='index')
train_dataset, valid_dataset, test_dataset = delaney_datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

# Do setup required for tf/keras models
# Number of features on conv-mols
n_feat = 75
# Batch size of models
batch_size = 128
model = GraphConvModel(
    len(delaney_tasks), batch_size=batch_size, mode='regression')

# Fit trained model
# test
losses = []
for i in range(20):
  loss = model.fit(train_dataset, nb_epoch=1)
  print("loss: %s" % str(loss))
  losses.append(loss)
print("losses")
print(losses)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

plt.plot(losses)
plt.show()