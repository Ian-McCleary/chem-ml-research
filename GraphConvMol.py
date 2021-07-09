from rdkit import Chem
import deepchem as dc
import numpy as np
import pandas as pd
import tensorflow as tf

tf.random.set_seed(123)
np.random.seed(123)
# update task count as list ["task1", "task2"..]
loader = dc.data.CSVLoader(["task1"], feature_field="smiles", id_field="ids", featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False))
data = loader.create_dataset("dataset_10.csv")
transformer = dc.trans.NormalizationTransformer(transform_y=True, dataset=data)
dataset = transformer.transform(data)

# Splits dataset into train/validation/test
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=dataset)
task_count = len(train_dataset.y[0])

# metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
metric = dc.metrics.Metric(dc.metrics.rms_score)

# parameter optimization
params_dict = {
    'n_tasks': [task_count],
    'number_atom_features': [15, 30, 75, 100, 150],
    'graph_conv_layers': [[32, 32], [64,64], [128,128]],
    'dense_layer_size': [8, 16, 32, 64, 128],
    'dropouts': [0.1, 0.2, 0.5, 0.9],
    'learning_rate': [0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
}

optimizer = dc.hyper.GridHyperparamOpt(dc.models.GraphConvModel)
#transformers = [dc.trans.NormalizationTransformer(transform_y=True, dataset=data)]
best_model, best_hyperparams, all_results =  optimizer.hyperparam_search(params_dict, train_dataset, valid_dataset,
                                                                         metric, transformer)                                                                  
print(all_results)
print("\n")
print(best_model)
print("\n")
print(best_hyperparams)

# Single evaluation model
# Single task: params (1, 50, 0.2, 1e-06)
print("Hyperparam list")
print(best_hyperparams[1])
print(best_hyperparams[2])


# supports width of channels from convolutional layers and dropout for each layer.
model = dc.models.GraphConvModel(
  n_tasks=task_count,
  number_atom_features=best_hyperparams[1],
  graph_conv_layers=best_hyperparams[2],
  dense_layer_size=best_hyperparams[3],
  dropouts=best_hyperparams[4],
  learning_rate=best_hyperparams[5],
  mode="regression"
)

# Fit trained model
# test
losses = []
for i in range(20):
  loss = model.fit(train_dataset, nb_epoch=1)
  print("loss: %s" % str(loss))
  losses.append(loss)
print("losses")
print(losses)

losses = []
for i in range(20):
  loss = model.fit(valid_dataset, nb_epoch=1)
  print("loss: %s" % str(loss))
  losses.append(loss)
print("losses")
print(losses)

print("Panda coeff")
df = pd.DataFrame(data=[train_dataset.X,train_dataset.y])
print(df.corr())

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], [transformer])
valid_scores = model.evaluate(valid_dataset, [metric], [transformer])

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

