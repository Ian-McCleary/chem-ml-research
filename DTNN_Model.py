from keras import backend as K
import os
import tensorflow as tf

# https://www.tensorflow.org/api_docs/python/tf/config/threading/set_inter_op_parallelism_threads
# K.set_session(K.tf.compat.v1.Session(config=K.tf.compat.v1.ConfigProto(intra_op_parallelism_threads=int(os.environ['OMP_NUM_THREADS']), 
# inter_op_parallelism_threads=int(os.environ['OMP_NUM_THREADS']))))
tf.config.threading.set_inter_op_parallelism_threads(
    int(os.environ['OMP_NUM_THREADS'])
)

# config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=0, 
#                        inter_op_parallelism_threads=0,
#                        device_count = {'CPU': int(os.environ['OMP_NUM_THREADS'])})

# session = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(session)

from rdkit import Chem
import deepchem as dc
import numpy as np
import pandas as pd

# update task count as list ["task1", "task2"..]
# TODO check that transformers are applied
loader = dc.data.CSVLoader(["task1"], feature_field="smiles", id_field="ids", featurizer=dc.feat.CoulombMatrix(max_atoms=65))
data = loader.create_dataset("Datasets/dataset_1000.csv")

# Splits dataset into train/validation/test
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=data)
task_count = len(train_dataset.y[0])


metrics = [
    dc.metrics.Metric(dc.metrics.rms_score)
    #dc.metrics.Metric(dc.metrics.r2_score)
    ]
metric = dc.metrics.Metric(dc.metrics.rms_score)
'''
# parameter optimization
params_dict = {
    'n_tasks': [task_count],
    'n_embedding': [5, 10, 50, 100],
    'dropouts': [0.2, 0.5],
    'learning_rate': [0.001, 0.0001]
}

optimizer = dc.hyper.GridHyperparamOpt(dc.models.DTNNModel)
transformers = [dc.trans.NormalizationTransformer(transform_y=True, dataset=data)]
best_model, best_hyperparams, all_results =  optimizer.hyperparam_search(params_dict, train_dataset, valid_dataset,
                                                                         metric, transformers)                                                                  
print(all_results)
print("\n")
print(best_hyperparams)

# Single evaluation model
# Single task: params (1, 50, 0.2, 1e-06)
print("Hyperparam list")
print(best_hyperparams[1])
print(best_hyperparams[2])

model = dc.models.DTNNModel(
    n_tasks=task_count,
    n_embedding=best_hyperparams[1],
    mode="regression",
    dropout=best_hyperparams[2],
    learning_rate=best_hyperparams[3]
)
'''

model = dc.models.DTNNModel(
    n_tasks = task_count,
    n_embedding=50,
    mode="regression",
    dropout=0.2,
    learning_rate=0.001
)
# Fit trained model
# test
train_losses = []
for i in range(500):
  loss = model.fit(train_dataset, nb_epoch=1)
  print("loss: %s" % str(loss))
  train_losses.append(loss)
print("losses")
print(train_losses)
print("\n")
print("Valid_dataset losses:")

valid_losses = []
for i in range(500):
    loss = model.fit(valid_dataset, nb_epoch=1)
    print("loss: %s" % str(loss))
    valid_losses.append(loss)
print("losses")
print(valid_losses)

df = pd.DataFrame(list(zip(train_losses, valid_losses)), columns=["train_losses", "valid_losses"])
df.to_csv("DTNN_fixed_learn_loss2.csv")

# model.fit(train_dataset)
# How well the model fits the training subset of our data
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
