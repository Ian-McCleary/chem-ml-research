from rdkit import Chem
import deepchem as dc
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse

# Driver method
def start_training():
      
  tf.random.set_seed(123)
  np.random.seed(123)
  # update task count as list ["task1", "task2"..]
  loader = dc.data.CSVLoader(["task1"], feature_field="smiles", id_field="ids", featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False))
  data = loader.create_dataset("Datasets/dataset_10000.csv")
  transformer = dc.trans.NormalizationTransformer(transform_y=True, dataset=data)
  dataset = transformer.transform(data)

  # Splits dataset into train/validation/test
  splitter = dc.splits.RandomSplitter()
  train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=dataset)
  task_count = len(train_dataset.y[0])

  # metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
  metric = dc.metrics.Metric(dc.metrics.rms_score)

  # model = param_optimization(train_dataset, valid_dataset, test_dataset, task_count, metric, transformer)
  # model = fixed_param_model(task_count)
  # loss_over_epoch(model, train_dataset, valid_dataset, metric, transformer)
  find_learn_rate(task_count,valid_dataset)


  # parameter optimization
  '''
  params_dict = {
      'n_tasks': [task_count],
      'number_atom_features': [15, 30, 75, 100, 150],
      'graph_conv_layers': [[32, 32], [64,64], [128,128]],
      'dense_layer_size': [8, 16, 32, 64, 128],
      'dropouts': [0.1, 0.2, 0.5, 0.9],
      'learning_rate': [0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
      'mode': ["regression"],
  }
  '''
def param_optimization(train_dataset, valid_dataset, test_dataset, task_count, metric, transformer):
  params_dict = {
      'n_tasks': [task_count],
      'number_atom_features': [75, 100, 150],
      'graph_conv_layers': [[32, 32], [64,64]],
      'dense_layer_size': [64, 128],
      'dropouts': [0.2, 0.5],
      'learning_rate': [0.001, 0.0001],
      'mode': ["regression"],
  }

  optimizer = dc.hyper.GridHyperparamOpt(dc.models.GraphConvModel)
  #transformers = [dc.trans.NormalizationTransformer(transform_y=True, dataset=data)]
  best_model, best_hyperparams, all_results =  optimizer.hyperparam_search(params_dict, train_dataset, valid_dataset,
                                                                          metric, [transformer])                                                                  
  print(all_results)
  print("\n")
  print(best_hyperparams)

  # Single evaluation model
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
  return model


def fixed_param_model(task_count):
  l_rate = 0.1
  model = dc.models.GraphConvModel(
    n_tasks=task_count,
    number_atom_features=100,
    dense_layer_size=128,
    graph_conv_layers=[32, 32],
    dropouts=0.2,
    learning_rate=l_rate,
    mode="regression"
  )
  return model

def find_learn_rate(task_count, valid_dataset):
  l_rate = 0.00001
  learn_arr = []
  loss_arr = []
  while l_rate < 0.1:
    model = dc.models.GraphConvModel(
      n_tasks=task_count,
      number_atom_features=100,
      dense_layer_size=128,
      graph_conv_layers=[32, 32],
      dropouts=0.2,
      learning_rate=l_rate,
      mode="regression"
    )
    loss = model.fit(valid_dataset, nb_epoch=5)
    loss_arr.append(loss)
    learn_arr.append(l_rate)
    l_rate = l_rate * 1.2

  df = pd.DataFrame(list(zip(learn_arr, loss_arr)), columns=["learning_rate", "validity_loss"])
  df.to_csv("gcm_learning_curve2.csv")

# Calculate loss over multiple training rounds
def loss_over_epoch(model, train_dataset, valid_dataset, metric, transformer):
  # Fit trained model
  train_losses = []
  for i in range(100):
    loss = model.fit(train_dataset, nb_epoch=1)
    print("loss: %s" % str(loss))
    train_losses.append(loss)
  print("losses")
  print(train_losses)

  valid_losses = []
  for i in range(100):
    loss = model.fit(valid_dataset, nb_epoch=1)
    print("loss: %s" % str(loss))
    valid_losses.append(loss)
  print("losses")
  print(valid_losses)

  # file_name = "loss_" + str(l_rate) + ".csv"
  df = pd.DataFrame(list(zip(train_losses, valid_losses)), columns=["train_losses", "valid_losses"])
  df.to_csv("loss_output.csv")

  print("Evaluating model")
  train_scores = model.evaluate(train_dataset, [metric], [transformer])
  valid_scores = model.evaluate(valid_dataset, [metric], [transformer])

  print("Train scores")
  print(train_scores)

  print("Validation scores")
  print(valid_scores)
  # l_rate = l_rate * 0.1
'''
def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("learn_rate",
                      help="Specify the learning rate of the model",
                      type=float,
                      default=0.001
                      )
  return parser.parse_args()
'''

def main():
  # args = parse_args()
  start_training()


if __name__ == "__main__":
  main()

