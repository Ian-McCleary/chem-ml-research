import tensorflow as tf
import os

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=int(os.environ['OMP_NUM_THREADS']), 
                        inter_op_parallelism_threads=int(os.environ['OMP_NUM_THREADS']),
                        device_count = {'CPU': int(os.environ['OMP_NUM_THREADS'])})

session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

from rdkit import Chem
import deepchem as dc
import numpy as np
import pandas as pd
from random import randrange

train_arr = []
valid_arr = []
test_arr = []
score_list = []


def k_fold_validation(model):
  for i in range(50):    
    # Set the seed
    dataseed = randrange(1000)
    np.random.seed(dataseed)
    tf.random.set_seed(dataseed)
    loader = dc.data.CSVLoader(["task1"], feature_field="smiles", id_field="ids", featurizer=dc.feat.CircularFingerprint(size=2048, radius=2))
    data = loader.create_dataset("Datasets/dataset_3task_1000.csv")

    transformer = dc.trans.NormalizationTransformer(dataset=data, transform_y=True)
    dataset = transformer.transform(data)

    # Splits dataset into train/validation/test
    splitter = dc.splits.RandomSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=dataset, seed=dataseed)
    task_count = len(dataset.y[0])
    n_features = len(dataset.X[0])

    print(task_count)
    #tasks, datasets, transformers = dc.molnet.load_hiv(featurizer='ECFP', split='scaffold')
    #train_dataset, valid_dataset, test_dataset = datasets
    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)

    model.fit(train_dataset, nb_epoch=100)
    # How well the model fits the training subset of our data
    train_scores = model.evaluate(train_dataset, metric)
    # Validation of the model over several training iterations.
    valid_score = model.evaluate(valid_dataset, metric)
    # How well the model generalizes the rest of the data
    test_score = model.evaluate(test_dataset, metric)
    train_arr.append(train_scores)
    valid_arr.append(valid_score)
    test_arr.append(test_score)
    '''
    # Fit trained model
    # test
    train_losses = []
    for i in range(300):
      loss = model.fit(train_dataset, nb_epoch=1)
      print("loss: %s" % str(loss))
      train_losses.append(loss)
    print("losses")
    print(train_losses)
    print("\n")
    print("Valid_dataset losses:")
    '''



def hyperparameter_optimization():
  dataseed = randrange(1000)
  np.random.seed(dataseed)
  tf.random.set_seed(dataseed)
  loader = dc.data.CSVLoader(["task1", "task2", "task3"], feature_field="smiles", id_field="ids", featurizer=dc.feat.CircularFingerprint(size=2048, radius=2))
  data = loader.create_dataset("Datasets/dataset_3task_1000.csv")

  transformer = dc.trans.NormalizationTransformer(dataset=data, transform_y=True)
  dataset = transformer.transform(data)

  # Splits dataset into train/validation/test
  splitter = dc.splits.RandomSplitter()
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=dataset, frac_train=0.8, frac_valid=0.2, seed=dataseed)
  task_count = len(dataset.y[0])
  n_features = len(dataset.X[0])

  params_dict = {
      'n_tasks': [task_count],
      'n_features': [n_features],
      'layer_sizes': [[256, 512, 1024], [1024, 1024, 1024], [1024, 512, 256]],
      'dropouts': [0.1, 0.2, 0.5],
      'learning_rate': [0.001, 0.0001, 0.00001]
  }
  #print(data.y)


  optimizer = dc.hyper.GridHyperparamOpt(dc.models.MultitaskRegressor)
  metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)
  best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
          params_dict, train_dataset, valid_dataset, metric, [transformer])
  print(best_hyperparams)
  train_loss(best_model, train_dataset=train_dataset, valid_dataset=valid_dataset, metric=metric, transformer=[transformer])



def train_loss(model, train_dataset, valid_dataset, metric, transformer):
  train_losses = []
  valid_eval = []
  all_loss = []
  for i in range(250):
    loss = model.fit(train_dataset, nb_epoch=1)
    valid = model.evaluate(valid_dataset, metric, transformer)
    train = model.evaluate(train_dataset, metric, transformer)
    print("loss: %s" % str(loss))
    train_losses.append(train)
    valid_eval.append(valid)
  all_loss.append(train_losses)
  all_loss.append(valid_eval)
  df = pd.DataFrame(list(zip(train_losses, valid_eval)), columns=["train_scores", "valid_scores"])
  df.to_csv("mtr_optimized2.csv")
  return all_loss
    
'''
evaluations = []
for i in range(3):
  dataseed = randrange(1000)
  np.random.seed(dataseed)
  tf.random.set_seed(dataseed)
  loader = dc.data.CSVLoader(["task1"], feature_field="smiles", id_field="ids", featurizer=dc.feat.CircularFingerprint(size=2048, radius=2))
  data = loader.create_dataset("Datasets/dataset_3task_1000.csv")

  transformer = dc.trans.NormalizationTransformer(dataset=data, transform_y=True)
  dataset = transformer.transform(data)

  # Splits dataset into train/validation/test
  splitter = dc.splits.RandomSplitter()
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=dataset, seed=dataseed)
  task_count = len(dataset.y[0])
  n_features = len(dataset.X[0])

  metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)

  # model = hyperparameter_optimization()
  #k_fold_validation(model)
  model = dc.models.MultitaskRegressor(
      n_tasks=task_count,
      n_features=n_features,
      layer_sizes=[1000, 1000, 1000, 1000],
      weight_decay_penalty_type ="l2",
      dropouts=0.2,
      learning_rate=0.0001,
      mode="regression"
    )
  both_list = train_loss(model, train_dataset, valid_dataset, metric, [transformer])
  evaluations.append(both_list[0])
  evaluations.append(both_list[1])

'''
hyperparameter_optimization()
#file_name = "mtr_loss_5.csv"
#df = pd.DataFrame(list(zip(evaluations[0], evaluations[1], evaluations[2], evaluations[3], evaluations[4], evaluations[5])), columns=[
#  "train_scores_1", "valid_scores_1","train_scores_2", "valid_scores_2","train_scores_3", "valid_scores_3"])
#df = pd.DataFrame(list(zip(all_loss)), columns=["all_loss"])
#df.to_csv(file_name) 
#file_name = "mtr_sensitivity_testing.csv"
# df = pd.DataFrame(list(zip(train_losses, valid_losses, train, valid, test)), columns=["train_losses", "valid_losses", "train_score", "valid_score", "test_score"])
#df = pd.DataFrame(list(zip(train_arr, valid_arr, test_arr)), columns=["train_scores", "valid_scores", "test_scores",])
#df.to_csv(file_name)