from random import randrange
import argparse
import pandas as pd
import numpy as np
import deepchem as dc
from rdkit import Chem
import tensorflow as tf
import os

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=int(os.environ['OMP_NUM_THREADS']),
                                  inter_op_parallelism_threads=int(
                                      os.environ['OMP_NUM_THREADS']),
                                  device_count={'CPU': int(os.environ['OMP_NUM_THREADS'])})

session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# Set the seed
dataseed = 8675309
np.random.seed(dataseed)
tf.random.set_seed(dataseed)

# Driver method


def start_training():
    
    # update task count as list ["task1", "task2"..]
    loader = dc.data.CSVLoader(["task1", "task2", "task3"], feature_field="smiles",
                               id_field="ids", featurizer=dc.feat.ConvMolFeaturizer())
    data = loader.create_dataset("Datasets/dataset_50k_3task.csv")
    transformer = dc.trans.NormalizationTransformer(
        transform_y=True, dataset=data)
    dataset = transformer.transform(data)

    # Splits dataset into train/validation/test
    splitter = dc.splits.RandomSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset=dataset, seed=dataseed)
    task_count = len(train_dataset.y[0])

    # metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)

    #model = param_optimization(
    #    train_dataset, valid_dataset, test_dataset, task_count, metric, transformer)
    model = fixed_param_model(task_count)
    all_loss = loss_over_epoch(model, train_dataset, valid_dataset, metric, transformer)
    file_name = "gc_50k_hyper.csv"
    df = pd.DataFrame(list(
        zip(all_loss[0], all_loss[1], all_loss[2], all_loss[3], all_loss[4], all_loss[5], all_loss[6], all_loss[7])),
                      columns=[
                          "train_mean", "train_eiso", "train_riso", "train_vert", "valid_mean", "valid_eiso",
                          "valid_riso", "valid_vert"])

    df.to_csv(file_name)


def k_fold_cross_validation():
    # Set the seed
    dataseed = randrange(1000)
    np.random.seed(dataseed)
    tf.random.set_seed(dataseed)
    # update task count as list ["task1", "task2"..]
    loader = dc.data.CSVLoader(["task1"], feature_field="smiles", id_field="ids",
                               featurizer=dc.feat.ConvMolFeaturizer(per_atom_fragmentation=False))
    data = loader.create_dataset("Datasets/dataset_1000.csv")
    transformer = dc.trans.NormalizationTransformer(
        transform_y=True, dataset=data)
    dataset = transformer.transform(data)

    # Splits dataset into train/validation/test
    splitter = dc.splits.RandomSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset=dataset, seed=dataseed)
    task_count = len(train_dataset.y[0])

    # metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)
    model = dc.models.GraphConvModel(
        n_tasks=task_count,
        number_atom_features=100,
        dense_layer_size=128,
        graph_conv_layers=[64, 64],
        dropouts=0.2,
        learning_rate=0.001,
        mode="regression"
    )


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
        'number_atom_features': [50, 75, 100, 150],
        'graph_conv_layers': [[32, 32], [64, 64], [128, 128]],
        'dense_layer_size': [16, 32, 64, 128],
        'dropouts': [0.2, 0.5, 0.1, 0.7],
        'learning_rate': [0.001, 0.0001, 0.00001],
        'mode': ["regression"],
    }
    optimizer = dc.hyper.GridHyperparamOpt(dc.models.GraphConvModel)
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(params_dict, train_dataset, valid_dataset,
                                                                            metric, [transformer])
    print(best_hyperparams)
    return best_model

# Parameter optimized for 10k, 1 task
def fixed_param_model(task_count):
    #l_rate = 0.00001
    l_rate = dc.models.optimizers.ExponentialDecay(0.0002, 0.9, 100)
    model = dc.models.GraphConvModel(
        n_tasks=task_count,
        number_atom_features=75,
        dense_layer_size=16,
        graph_conv_layers=[32, 32],
        dropouts=0.5,
        learning_rate=l_rate,
        mode="regression"
    )
    return model


def find_learn_rate(task_count, train_dataset):
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
        loss = model.fit(train_dataset, nb_epoch=5)
        loss_arr.append(loss)
        learn_arr.append(l_rate)
        l_rate = l_rate + 0.0002

    df = pd.DataFrame(list(zip(learn_arr, loss_arr)), columns=[
                      "learning_rate", "training_loss"])
    df.to_csv("gcm_learning_curve3.csv")

# Calculate loss over multiple training rounds


def loss_over_epoch(model, train_dataset, valid_dataset, metric, transformer):
    train_mean = []
    train_eiso = []
    train_riso = []
    train_vert = []

    valid_mean = []
    valid_eiso = []
    valid_riso = []
    valid_vert = []
    all_loss = []

    for i in range(1000):
        loss = model.fit(train_dataset, nb_epoch=1)
        train = model.evaluate(train_dataset, metric, transformer, per_task_metrics=True)
        valid = model.evaluate(valid_dataset, metric, transformer, per_task_metrics=True)
        print("loss: %s" % str(loss))
        print(type(valid))
        print(valid)
        # print(valid[0]["mean_absolute_error"])
        # print(valid[1]["mean_absolute_error"])
        # print(valid[1]["mean_absolute_error"][0])
        train_mean.append(train[0]["mean_absolute_error"])
        train_eiso.append(train[1]["mean_absolute_error"][0])
        train_riso.append(train[1]["mean_absolute_error"][1])
        train_vert.append(train[1]["mean_absolute_error"][2])

        valid_mean.append(valid[0]["mean_absolute_error"])
        valid_eiso.append(valid[1]["mean_absolute_error"][0])
        valid_riso.append(valid[1]["mean_absolute_error"][1])
        valid_vert.append(valid[1]["mean_absolute_error"][2])
    # all_loss.extend([train_mean, train_eiso, train_riso, train_vert])
    # all_loss.extend([valid_mean, valid_eiso, valid_riso, valid_vert])
    all_loss.append(train_mean)
    all_loss.append(train_eiso)
    all_loss.append(train_riso)
    all_loss.append(train_vert)

    all_loss.append(valid_mean)
    all_loss.append(valid_eiso)
    all_loss.append(valid_riso)
    all_loss.append(valid_vert)
    print(len(all_loss))
    return all_loss

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
