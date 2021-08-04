import os
import time
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

def start_training():

    loader = dc.data.CSVLoader(["task1", "task2", "task3"], feature_field="smiles", id_field="ids", featurizer=dc.feat.CoulombMatrix(max_atoms=70))
    data = loader.create_dataset("Datasets/dataset_3task_10k_filtered.csv")
    transformer = dc.trans.NormalizationTransformer(
        dataset=data, transform_y=True)
    dataset = transformer.transform(data)

    # Splits dataset into train/validation/test
    splitter = dc.splits.RandomSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=dataset, frac_train=0.70, frac_valid=0.15, frac_test=0.15, seed=dataseed)
    task_count = len(train_dataset.y[0])

    metric = dc.metrics.Metric(dc.metrics.rms_score)
    metrics = [dc.metrics.Metric(dc.metrics.rms_score), dc.metrics.Metric(dc.metrics.r2_score)]
    #model = dtnn_fixed_param_model(task_count=task_count)
    model = dtnn_hyperparameter_optimization(train_dataset, valid_dataset, transformer, metric)
    all_loss = loss_over_epoch(model, train_dataset, valid_dataset, test_dataset, metrics, [transformer])
    k_fold_validation(model, data)
    df = pd.DataFrame(list(zip(all_loss[0], all_loss[1], all_loss[2], all_loss[3], all_loss[4], all_loss[5], all_loss[6], all_loss[7])),columns=[
                          "train_mean", "train_eiso", "train_riso", "train_vert", "valid_mean", "valid_eiso",
                          "valid_riso", "valid_vert"])
    df.to_csv("dtnn_10k_hyper_filtered.csv")

def dtnn_hyperparameter_optimization(train_dataset, valid_dataset, transformer, metric):
    # parameter optimization
    task_count = len(train_dataset.y[0])
    params_dict = {
        'n_tasks': [task_count],
        'n_embedding': [10, 50, 100, 75],
        'dropouts': [0.2, 0.5, 0.75],
        'learning_rate': [0.001, 0.0001, 0.00001]
    }

    optimizer = dc.hyper.GridHyperparamOpt(dc.models.DTNNModel)
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(params_dict, train_dataset, valid_dataset,
                                                                            metric, [transformer])

    print(best_hyperparams)
    return best_model

def dtnn_fixed_param_model(task_count):
    model = dc.models.DTNNModel(
        n_tasks=task_count,
        n_embedding=50,
        distance_max=24,
        mode="regression",
        dropout=0.2,
        learning_rate=0.001
    )
    return model

def main():
    # args = parse_args()
    start_training()


if __name__ == "__main__":
    main()

