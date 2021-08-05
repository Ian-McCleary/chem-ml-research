import os
import time
from random import randrange

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


def loss_over_epoch(model, train_dataset, valid_dataset, test_dataset, metric, transformer, epochs):
    #metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
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
        # print(valid[0]["mean_absolute_error"])
        # print(valid[1]["mean_absolute_error"])
        # print(valid[1]["mean_absolute_error"][0])
        train_mean.append(train[0]["rms_score"])
        train_eiso.append(train[1]["rms_score"][0])
        train_riso.append(train[1]["rms_score"][1])
        train_vert.append(train[1]["rms_score"][2])

        valid_mean.append(valid[0]["rms_score"])
        valid_eiso.append(valid[1]["rms_score"][0])
        valid_riso.append(valid[1]["rms_score"][1])
        valid_vert.append(valid[1]["rms_score"][2])
    # all_loss.extend([train_mean, train_eiso, train_riso, train_vert])mean
    # all_loss.extend([valid_mean, valid_eiso, valid_riso, valid_vert])
    all_loss.append(train_mean)
    all_loss.append(train_eiso)
    all_loss.append(train_riso)
    all_loss.append(train_vert)

    all_loss.append(valid_mean)
    all_loss.append(valid_eiso)
    all_loss.append(valid_riso)
    all_loss.append(valid_vert)

    test_scores = model.evaluate(test_dataset, metric, [transformer], per_task_metrics=True)
    print("mean rms:")
    print(test_scores[0]["rms_score"])
    print("eiso rms:")
    print(test_scores[1]["rms_score"][0])
    print("riso rms:")
    print(test_scores[1]["rms_score"][1])
    print("vert rms:")
    print(test_scores[1]["rms_score"][2])
    print("\n")
    print("mean r2:")
    print(test_scores[0]["r2_score"])
    print("eiso r2:")
    print(test_scores[1]["r2_score"][0])
    print("riso r2:")
    print(test_scores[1]["r2_score"][1])
    print("vert r2:")
    print(test_scores[1]["r2_score"][2])
    return all_loss



def k_fold_validation(model, data):
    eiso_scores = []
    riso_scores = []
    vert_scores = []

    transformer = dc.trans.NormalizationTransformer(
        dataset=data, transform_y=True)
    dataset = transformer.transform(data)

    for i in range(50):
        # Set the seed
        dataseed = randrange(1000)
        np.random.seed(dataseed)
        tf.random.set_seed(dataseed)
        # Splits dataset into train/validation/test
        splitter = dc.splits.RandomSplitter()
        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
            dataset=dataset, frac_train=0.70, frac_valid=0.15, frac_test=0.15, seed=dataseed)
        task_count = len(dataset.y[0])
        n_features = len(dataset.X[0])
        # tasks, datasets, transformers = dc.molnet.load_hiv(featurizer='ECFP', split='scaffold')
        # train_dataset, valid_dataset, test_dataset = datasets
        metric = dc.metrics.Metric(dc.metrics.rms_score)

        model.fit(train_dataset, nb_epoch=50)
        test_score = model.evaluate(test_dataset, metric, [transformer], per_task_metrics=True)
        eiso_scores.append(test_score[1]["rms_score"][0])
        riso_scores.append(test_score[1]["rms_score"][1])
        vert_scores.append(test_score[1]["rms_score"][2])

    eiso_min, riso_min, vert_min = 1000, 1000, 1000
    eiso_max, riso_max, vert_max = -1000, -1000, -1000
    eiso_total, riso_total, vert_total = 0, 0, 0

    for x in range(len(eiso_scores)):
        eiso_total += eiso_scores[x]
        riso_total += riso_scores[x]
        vert_total += vert_scores[x]

        if eiso_scores[x] > eiso_max:
            eiso_max = eiso_scores[x]
        elif eiso_scores[x] < eiso_min:
            eiso_min = eiso_scores[x]

        if riso_scores[x] > riso_max:
            riso_max = riso_scores[x]
        elif riso_scores[x] < riso_min:
            riso_min = riso_scores[x]

        if vert_scores[x] > vert_max:
            vert_max = vert_scores[x]
        elif vert_scores[x] < vert_min:
            vert_min = vert_scores[x]
    eiso_total = eiso_total/50
    riso_total = riso_total/50
    vert_total = vert_total/50
    print("eiso:")
    print(eiso_total)
    print(eiso_max)
    print(eiso_min)
    print("riso:")
    print(riso_total)
    print(riso_max)
    print(riso_min)
    print("vert:")
    print(vert_total)
    print(vert_max)
    print(vert_min)



