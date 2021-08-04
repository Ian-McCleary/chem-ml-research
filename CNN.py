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

# https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=int(os.environ['OMP_NUM_THREADS']),
                                  inter_op_parallelism_threads=int(
                                      os.environ['OMP_NUM_THREADS']),
                                  device_count={'CPU': int(os.environ['OMP_NUM_THREADS'])})

session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def cnn_start_training():
    # Set the seed
    dataseed = 8675309
    np.random.seed(dataseed)
    tf.random.set_seed(dataseed)

    max_a = 70
    fp_len = 2048
    # Attempt to use 2 featurizers
    # This process takes a while, consider doing to full dataset.
    loader_cfp = dc.data.CSVLoader(["task1", "task2", "task3"], feature_field="smiles", id_field="ids",
                               featurizer=dc.feat.CircularFingerprint(size=fp_len, radius=2))
    data_cfp = loader_cfp.create_dataset("Datasets/dataset_3task_100.csv")

    loader_cm = dc.data.CSVLoader(["task1", "task2", "task3"], feature_field="smiles", id_field="ids",
                               featurizer=dc.feat.CoulombMatrix(max_atoms=max_a))
    data_cm = loader_cm.create_dataset("Datasets/dataset_3task_100.csv")

    # dual_feature_ds = dc.data.NumpyDataset(X=input_x, y=data_cm.y, ids=data_cm.ids, n_tasks=3)
    transformer = dc.trans.NormalizationTransformer(
        dataset=data_cm, transform_y=True)
    dataset = transformer.transform(data_cm)

    # Splits dataset into train/validation/test
    splitter = dc.splits.RandomSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset=dataset, frac_train=0.85, frac_valid=0.15, frac_test=0.00, seed=dataseed)
    task_count = len(dataset.y[0])
    n_features = 4900

    metric = dc.metrics.Metric(dc.metrics.r2_score)
    model = cnn_fixed_param_model(task_count, n_features)
    all_loss = loss_over_epoch(model, train_dataset, valid_dataset, metric, transformer)

    df = pd.DataFrame(list(
        zip(all_loss[0], all_loss[1], all_loss[2], all_loss[3], all_loss[4], all_loss[5], all_loss[6], all_loss[7])),
        columns=[
            "train_mean", "train_eiso", "train_riso", "train_vert", "valid_mean", "valid_eiso",
            "valid_riso", "valid_vert"])
    df.to_csv("cnn_fixed_param2.csv")


def cnn_fixed_param_model(n_tasks, n_features):
    model = dc.models.CNN(
        n_tasks,
        n_features,
        dims=2,
        layer_filters=[20, 20],
        kernel_size=[3, 3],
        weight_init_stddevs=0.02,
        bias_init_consts=1,
        weight_decay_penalty=1.3,
        weight_decay_penalty_type="l2",
        dropouts=0.25,
        mode="regression"

    )
    return model


def main():
    # args = parse_args()
    cnn_start_training()


if __name__ == "__main__":
    main()
