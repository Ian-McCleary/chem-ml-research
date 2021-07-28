import time
from random import randrange
import pandas as pd
import numpy as np
import deepchem as dc
from rdkit import Chem
import os
import tensorflow as tf

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
    data = loader.create_dataset("Datasets/dataset_50k_3task.csv")
    transformer = dc.trans.NormalizationTransformer(
        dataset=data, transform_y=True)
    dataset = transformer.transform(data)

    # Splits dataset into train/validation/test
    splitter = dc.splits.RandomSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=dataset, seed=dataseed)
    task_count = len(train_dataset.y[0])

    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)
    #model = fixed_param_model(task_count=task_count)
    model = hyperparameter_optimization(train_dataset, valid_dataset, transformer, metric)
    all_loss = train_loss_over_epoch(model, train_dataset, valid_dataset, metric, [transformer])

    df = pd.DataFrame(list(zip(all_loss[0], all_loss[1], all_loss[2], all_loss[3], all_loss[4], all_loss[5], all_loss[6], all_loss[7])),columns=[
                          "train_mean", "train_eiso", "train_riso", "train_vert", "valid_mean", "valid_eiso",
                          "valid_riso", "valid_vert"])
    df.to_csv("dtnn_50k_hyper.csv")

def hyperparameter_optimization(train_dataset, valid_dataset, transformer, metric):
    # parameter optimization
    task_count = len(train_dataset.y[0])
    params_dict = {
        'n_tasks': [task_count],
        'n_embedding': [10, 50, 100],
        'dropouts': [0.2, 0.5],
        'learning_rate': [0.001, 0.0001, 0.00001]
    }

    optimizer = dc.hyper.GridHyperparamOpt(dc.models.DTNNModel)
    best_model, best_hyperparams, all_results =  optimizer.hyperparam_search(params_dict, train_dataset, valid_dataset,
                                                                            metric, [transformer])
    print(all_results)
    print("\n")
    print(best_hyperparams)
    return best_model

def fixed_param_model(task_count):
    model = dc.models.DTNNModel(
        n_tasks=task_count,
        n_embedding=50,
        distance_max=24,
        mode="regression",
        dropout=0.2,
        learning_rate=0.001
    )
    return model


def train_loss_over_epoch(model, train_dataset, valid_dataset, metric, transformer):
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
    
    #df = pd.DataFrame(list(zip(train_mean, valid_losses)),
     #               columns=["train_losses", "valid_losses"])
    #df.to_csv("DTNN_3task_seperate_metric.csv")

def main():
    # args = parse_args()
    start_training()


if __name__ == "__main__":
    main()

