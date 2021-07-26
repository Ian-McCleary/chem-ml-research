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
dataseed = randrange(1000)
np.random.seed(dataseed)
tf.random.set_seed(dataseed)
def start_training():

    times = []
    for i in range(6):
        if i == 0:
            start_time = time.time()
            loader = dc.data.CSVLoader(["task1"], feature_field="smiles",
                                    id_field="ids", featurizer=dc.feat.CoulombMatrix(max_atoms=70))
            data = loader.create_dataset("Datasets/dataset_500.csv")
        if i == 1:
            start_time = time.time()
            loader = dc.data.CSVLoader(["task1"], feature_field="smiles",
                                    id_field="ids", featurizer=dc.feat.CoulombMatrix(max_atoms=70))
            data = loader.create_dataset("Datasets/dataset_1000.csv")
        if i == 2:
            start_time = time.time()
            loader = dc.data.CSVLoader(["task1"], feature_field="smiles",
                                    id_field="ids", featurizer=dc.feat.CoulombMatrix(max_atoms=70))
            data = loader.create_dataset("Datasets/dataset_5000.csv")
        if i == 3:
            start_time = time.time()
            loader = dc.data.CSVLoader(["task1"], feature_field="smiles",
                                    id_field="ids", featurizer=dc.feat.CoulombMatrix(max_atoms=70))
            data = loader.create_dataset("Datasets/dataset_7000.csv")
        if i == 4:
            start_time = time.time()
            loader = dc.data.CSVLoader(["task1"], feature_field="smiles",
                                    id_field="ids", featurizer=dc.feat.CoulombMatrix(max_atoms=70))
            data = loader.create_dataset("Datasets/dataset_10000.csv")
        if i == 5:
            start_time = time.time()
            loader = dc.data.CSVLoader(["task1"], feature_field="smiles",
                                    id_field="ids", featurizer=dc.feat.CoulombMatrix(max_atoms=70))
            data = loader.create_dataset("Datasets/dataset_15000.csv")

        transformer = dc.trans.NormalizationTransformer(
            dataset=data, transform_y=True)
        dataset = transformer.transform(data)

        # Splits dataset into train/validation/test
        splitter = dc.splits.RandomSplitter()
        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
            dataset=dataset, seed=dataseed)
        task_count = len(train_dataset.y[0])

        metrics = [
            dc.metrics.Metric(dc.metrics.mean_absolute_error)
            # dc.metrics.Metric(dc.metrics.r2_score)
        ]
        metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)
        model = fixed_param_model(task_count=task_count)
        loss_list = train_loss_over_epoch(model, train_dataset=train_dataset, valid_dataset=valid_dataset, metric=metric, transformer=transformer)
        end_time = time.time()
        time_elapsed = end_time-start_time
        times.append(time_elapsed)

    df = pd.DataFrame(list(zip(times)),
                    columns=["times(sec)"])
    df.to_csv("dtnn_time_complexity.csv")

def hyperparameter_optimization(train_dataset, valid_dataset, transformer, metric):
    # parameter optimization
    task_count = len(train_dataset.y[0])
    params_dict = {
        'n_tasks': [task_count],
        'n_embedding': [5, 10, 50, 100],
        'dropouts': [0.2, 0.5],
        'learning_rate': [0.001, 0.0001]
    }

    optimizer = dc.hyper.GridHyperparamOpt(dc.models.DTNNModel)
    best_model, best_hyperparams, all_results =  optimizer.hyperparam_search(params_dict, train_dataset, valid_dataset,
                                                                            metric, transformer)                                                                  
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


def train_loss_over_epoch(model, train_dataset, valid_dataset, metric, transformer):
    # Fit trained model
    # test
    valid_losses = []
    train_losses = []
    all_losses = []
    for i in range(250):
        loss = model.fit(train_dataset, nb_epoch=1)
        valid = model.evaluate(valid_dataset, metric, [
                            transformer], per_task_metrics=False)
        train = model.evaluate(train_dataset, metric, [
                            transformer], per_task_metrics=False)
        print("loss: %s" % str(loss))
        # train_mean.append(train[0])
        # train_task1.append(train)
        train_losses.append(train)
        valid_losses.append(valid)
    all_losses.append(train_losses)
    all_losses.append(valid_losses)
    return all_losses
    
    #df = pd.DataFrame(list(zip(train_mean, valid_losses)),
     #               columns=["train_losses", "valid_losses"])
    #df.to_csv("DTNN_3task_seperate_metric.csv")

def main():
    # args = parse_args()
    start_training()


if __name__ == "__main__":
    main()

