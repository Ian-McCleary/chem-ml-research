import argparse
import os
from random import randrange

from training_methods import loss_over_epoch
from training_methods import k_fold_validation

import deepchem as dc
import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem

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
    data = loader.create_dataset("Datasets/dataset_3task_50k_filtered.csv")
    transformer = dc.trans.NormalizationTransformer(transform_y=True, dataset=data)
    dataset = transformer.transform(data)

    # Splits dataset into train/validation/test
    splitter = dc.splits.RandomSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset=dataset,frac_train=0.70, frac_valid=0.15, frac_test=0.15, seed=dataseed)
    task_count = len(train_dataset.y[0])

    # metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
    metric = dc.metrics.Metric(dc.metrics.rms_score)
    metrics = [dc.metrics.Metric(dc.metrics.rms_score), dc.metrics.Metric(dc.metrics.r2_score)]

    model = param_optimization(
        train_dataset, valid_dataset, test_dataset, task_count, metric, transformer)
    #model = fixed_param_model(task_count)
    all_loss = loss_over_epoch(model, train_dataset, valid_dataset, test_dataset, metrics, transformer, 100)
    k_fold_validation(model, data)
    file_name = "gc_10k_hyper_filtered.csv"
    df = pd.DataFrame(list(
        zip(all_loss[0], all_loss[1], all_loss[2], all_loss[3], all_loss[4], all_loss[5], all_loss[6], all_loss[7])),
                      columns=[
                          "train_mean", "train_eiso", "train_riso", "train_vert", "valid_mean", "valid_eiso",
                          "valid_riso", "valid_vert"])

    df.to_csv(file_name)


def param_optimization(train_dataset, valid_dataset, test_dataset, task_count, metric, transformer):
    params_dict = {
        'n_tasks': [task_count],
        'number_atom_features': [75],
        'graph_conv_layers': [[64, 64]],
        'dense_layer_size': [128, 256],
        'dropouts': [0.5],
        'learning_rate': [0.0001, 0.00001],
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


def main():
    # args = parse_args()
    start_training()


if __name__ == "__main__":
    main()
