from random import randrange
import pandas as pd
import numpy as np

from training_methods import loss_over_epoch
from training_methods import k_fold_validation

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


train_arr = []
valid_arr = []
test_arr = []
score_list = []

# Driver function
def start_training():
    evaluations = []
    # dataseed = randrange(1000)
    dataseed = 8675309
    np.random.seed(dataseed)
    tf.random.set_seed(dataseed)
    loader = dc.data.CSVLoader(["task1", "task2", "task3"], feature_field="smiles", id_field="ids",
                                featurizer=dc.feat.CircularFingerprint(size=2048, radius=2))
    data = loader.create_dataset("Datasets/dataset_3task_50k_filtered.csv")
    
    transformer = dc.trans.NormalizationTransformer(
        dataset=data, transform_y=True)
    dataset = transformer.transform(data)

    # Splits dataset into train/validation/test
    splitter = dc.splits.RandomSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset=dataset, frac_train=0.70, frac_valid=0.15, frac_test=0.15, seed=dataseed)
    task_count = len(dataset.y[0])
    n_features = len(dataset.X[0])

    metric = dc.metrics.Metric(dc.metrics.rms_score)
    metrics = [dc.metrics.Metric(dc.metrics.rms_score), dc.metrics.Metric(dc.metrics.r2_score)]

    #model = mtr_fixed_param_model(task_count=task_count, n_features=n_features)
    model = mtr_hyperparameter_optimization(train_dataset, valid_dataset, transformer, metric)
    all_loss = loss_over_epoch(model, train_dataset, valid_dataset, test_dataset, metrics, transformer)
    k_fold_validation(model, data)
    # hyperparameter_optimization()
    file_name = "mtr_50k_hyper_filtered.csv"
    df = pd.DataFrame(list(zip(all_loss[0], all_loss[1], all_loss[2], all_loss[3], all_loss[4], all_loss[5], all_loss[6], all_loss[7])), columns=[
        "train_mean", "train_eiso", "train_riso", "train_vert", "valid_mean", "valid_eiso", "valid_riso", "valid_vert"])

    df.to_csv(file_name)

def mtr_hyperparameter_optimization(train_dataset, valid_dataset, transformer, metric):
    task_count = len(train_dataset.y[0])
    n_features = len(train_dataset.X[0])
    l_rate_scheduler = dc.models.optimizers.ExponentialDecay(0.0002, 0.9, 15)
    params_dict = {
        'n_tasks': [task_count],
        'n_features': [n_features],
        'layer_sizes': [[256, 512, 1024], [128, 256, 512], [64, 128, 256]],
        'dropouts': [0.1, 0.2, 0.5, 0.4, 0.6],
        'learning_rate': [0.001, 0.0001, 0.00001, l_rate_scheduler]
    }

    optimizer = dc.hyper.GridHyperparamOpt(dc.models.MultitaskRegressor)
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
        params_dict, train_dataset, valid_dataset, metric, [transformer])
    print(best_hyperparams)
    return best_model


def find_learn_rate(task_count, train_dataset):
    l_rate = 0.00001
    learn_arr = []
    loss_arr = []
    while l_rate < 0.1:
        model = dc.models.MultitaskRegressor(
            n_tasks=task_count,
            n_features=n_features,
            layer_sizes=[256, 512, 1024],
            weight_decay_penalty_type="l2",
            dropouts=0.1,
            learning_rate=l_rate,
            mode="regression"
        )
        loss = model.fit(train_dataset, nb_epoch=3)
        loss_arr.append(loss)
        learn_arr.append(l_rate)
        l_rate = l_rate + 0.0002

    df = pd.DataFrame(list(zip(learn_arr, loss_arr)), columns=[
                      "learning_rate", "training_loss"])
    df.to_csv("mtr_learning_curve.csv")


def mtr_fixed_param_model(task_count, n_features):
    model = dc.models.MultitaskRegressor(
        n_tasks=task_count,
        n_features=n_features,
        layer_sizes=[32, 64, 128],
        weight_decay_penalty_type="l2",
        dropouts=0.3,
        learning_rate=0.0001,
        mode="regression"
    )
    return model


def main():
    # args = parse_args()
    start_training()

if __name__ == "__main__":
    main()

