from random import randrange
import argparse
import pandas as pd
import numpy as np
import deepchem as dc
from rdkit import Chem
import tensorflow as tf
import os
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

    input_x = np.zeros((len(data_cm.X), 2, max_a, max_a))
    for i in range(len(input_x)):
        bit_to_mtrx = np.zeros((max_a, max_a))
        for y in range(max_a):
            for x in range(max_a):
                bit_pos = (max_a*y)+x
                if bit_pos < fp_len-1:
                    mtrx_val = data_cm.X[i][y][x]
                else:
                    mtrx_val = 0
                bit_to_mtrx[y][x] = mtrx_val
        single_cell = [data_cm.X[i], bit_to_mtrx]
        input_x[i] = single_cell
    print(input_x[0])
    print("\n")

    dual_feature_ds = dc.data.NumpyDataset(X=input_x, y=data_cm.y, ids=data_cm.ids, n_tasks=3)
    transformer = dc.trans.NormalizationTransformer(
        dataset=dual_feature_ds, transform_y=True)
    dataset = transformer.transform(dual_feature_ds)

    # Splits dataset into train/validation/test
    splitter = dc.splits.RandomSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset=dataset, frac_train=0.85, frac_valid=0.15, frac_test=0.00, seed=dataseed)
    task_count = len(dataset.y[0])
    n_features = (max_a * max_a)+fp_len

    metric = dc.metrics.Metric(dc.metrics.r2_score)
    model = cnn_fixed_param_model(task_count, n_features)
    all_loss = cnn_loss_over_epoch(model, train_dataset, valid_dataset, metric, transformer)

    df = pd.DataFrame(list(
        zip(all_loss[0], all_loss[1], all_loss[2], all_loss[3], all_loss[4], all_loss[5], all_loss[6], all_loss[7])),
        columns=[
            "train_mean", "train_eiso", "train_riso", "train_vert", "valid_mean", "valid_eiso",
            "valid_riso", "valid_vert"])
    df.to_csv("cnn_fixed_param2.csv")


def cnn_loss_over_epoch(model, train_dataset, valid_dataset, metric, transformer):
    train_mean = []
    train_eiso = []
    train_riso = []
    train_vert = []

    valid_mean = []
    valid_eiso = []
    valid_riso = []
    valid_vert = []
    all_loss = []

    for i in range(250):
        loss = model.fit(train_dataset, nb_epoch=1)
        train = model.evaluate(train_dataset, metric, [transformer], per_task_metrics=True)
        valid = model.evaluate(valid_dataset, metric, [transformer], per_task_metrics=True)
        print("loss: %s" % str(loss))
        print(type(valid))
        print(valid)
        # print(valid[0]["mean_absolute_error"])
        # print(valid[1]["mean_absolute_error"])
        # print(valid[1]["mean_absolute_error"][0])
        train_mean.append(train[0]["r2_score"])
        train_eiso.append(train[1]["r2_score"][0])
        train_riso.append(train[1]["r2_score"][1])
        train_vert.append(train[1]["r2_score"][2])

        valid_mean.append(valid[0]["r2_score"])
        valid_eiso.append(valid[1]["r2_score"][0])
        valid_riso.append(valid[1]["r2_score"][1])
        valid_vert.append(valid[1]["r2_score"][2])
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


def cnn_fixed_param_model(n_tasks, n_features):
    model = dc.models.CNN(
        n_tasks,
        n_features,
        dims=2,
        layer_filters=[20, 20],
        kernel_size=[1, 3, 3],
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