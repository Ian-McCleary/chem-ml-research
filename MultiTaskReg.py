from random import randrange
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
    data = loader.create_dataset("Datasets/dataset_50k_3task.csv")
    
    transformer = dc.trans.NormalizationTransformer(
        dataset=data, transform_y=True)
    dataset = transformer.transform(data)

    # Splits dataset into train/validation/test
    splitter = dc.splits.RandomSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset=dataset, frac_train=0.85, frac_valid=0.15, frac_test=0.00, seed=dataseed)
    task_count = len(dataset.y[0])
    n_features = len(dataset.X[0])

    metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)
    print("model \n")
    #model = fixed_param_model(task_count=task_count, n_features=n_features)
    model = hyperparameter_optimization(train_dataset, valid_dataset, transformer, metric)
    print("loss train \n")
    all_loss = train_loss(model, train_dataset, valid_dataset, metric, [transformer])
    print("csv: ")
    # hyperparameter_optimization()
    file_name = "Losses/mtr/mtr_3task_hyperparam_50k.csv"
    df = pd.DataFrame(list(zip(all_loss[0], all_loss[1], all_loss[2], all_loss[3], all_loss[4], all_loss[5], all_loss[6], all_loss[7])), columns=[
        "train_mean", "train_eiso", "train_riso", "train_vert", "valid_mean", "valid_eiso", "valid_riso", "valid_vert"])

    df.to_csv(file_name)



def k_fold_validation(model):
    for i in range(50):
        # Set the seed
        dataseed = randrange(1000)
        np.random.seed(dataseed)
        tf.random.set_seed(dataseed)
        loader = dc.data.CSVLoader(["task1"], feature_field="smiles", id_field="ids",
                                   featurizer=dc.feat.CircularFingerprint(size=2048, radius=2))
        data = loader.create_dataset("Datasets/dataset_3task_1000.csv")

        transformer = dc.trans.NormalizationTransformer(
            dataset=data, transform_y=True)
        dataset = transformer.transform(data)

        # Splits dataset into train/validation/test
        splitter = dc.splits.RandomSplitter()
        train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
            dataset=dataset, seed=dataseed)
        task_count = len(dataset.y[0])
        n_features = len(dataset.X[0])

        print(task_count)
        # tasks, datasets, transformers = dc.molnet.load_hiv(featurizer='ECFP', split='scaffold')
        # train_dataset, valid_dataset, test_dataset = datasets
        metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)

        model.fit(train_dataset, nb_epoch=100)
        # How well the model fits the training subset of our data
        train_scores = model.evaluate(train_dataset, metric)
        # Validation of the model over several training iterations.
        valid_score = model.evaluate(valid_dataset, metric)
        # How well the model generalizes the rest of the data
        test_score = model.evaluate(test_dataset, metric)
        train_arr.append(train_scores)
        valid_arr.append(valid_score)
        test_arr.append(test_score)


def hyperparameter_optimization(train_dataset, valid_dataset, transformer, metric):
    task_count = len(train_dataset.y[0])
    n_features = len(train_dataset.X[0])

    params_dict = {
        'n_tasks': [task_count],
        'n_features': [n_features],
        'layer_sizes': [[256, 512, 1024], [128, 256, 512], [64, 128, 256]],
        'dropouts': [0.1, 0.2, 0.5, 0.4, 0.6],
        'learning_rate': [0.001, 0.0001, 0.00001]
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

# train loss over epochs
# If per task metric is true, datatype = tuple(dict, dict{[array]})
def train_loss(model, train_dataset, valid_dataset, metric, transformer):
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
        train = model.evaluate(train_dataset, metric, transformer, per_task_metrics=True)
        valid = model.evaluate(valid_dataset, metric, transformer, per_task_metrics=True)
        print("loss: %s" % str(loss))
        print(type(valid))
        print(valid)
        #print(valid[0]["mean_absolute_error"])
        #print(valid[1]["mean_absolute_error"])
        #print(valid[1]["mean_absolute_error"][0])
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


def fixed_param_model(task_count, n_features):
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

'''
loader = dc.data.CSVLoader(["task1"], feature_field="smiles", id_field="ids",
                            featurizer=dc.feat.CircularFingerprint(size=2048, radius=2))
data = loader.create_dataset("Datasets/dataset_10000.csv")
 
transformer = dc.trans.NormalizationTransformer(
    dataset=data, transform_y=True)
dataset = transformer.transform(data)

# Splits dataset into train/validation/test
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
    dataset=dataset, frac_train=0.85, frac_valid=0.15, frac_test=0.00, seed=dataseed)
task_count = len(dataset.y[0])
n_features = len(dataset.X[0])

metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)

# model = hyperparameter_optimization()
# k_fold_validation(model)
model = None
model = dc.models.MultitaskRegressor(
    n_tasks=task_count,
    n_features=n_features,
    layer_sizes=[256, 512, 1024],
    weight_decay_penalty_type="l2",
    dropouts=0.1,
    learning_rate=0.0001,
    mode="regression"
)
both_list = train_loss(model, train_dataset,
                        valid_dataset, metric, [transformer])
evaluations.append(both_list[0])
evaluations.append(both_list[1])

dataseed = randrange(1000)
np.random.seed(dataseed)
tf.random.set_seed(dataseed)

loader = dc.data.CSVLoader(["task1"], feature_field="smiles", id_field="ids", featurizer=dc.feat.CircularFingerprint(size=2048, radius=2))
data = loader.create_dataset("Datasets/dataset_10000.csv")

transformer = dc.trans.NormalizationTransformer(dataset=data, transform_y=True)
dataset = transformer.transform(data)

splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=dataset,frac_train=0.85, frac_valid=0.15, frac_test=0.00, seed=dataseed)
task_count = len(dataset.y[0])
n_features = len(dataset.X[0])

metric = dc.metrics.Metric(dc.metrics.mean_absolute_error)

find_learn_rate(task_count=task_count, train_dataset=train_dataset)
'''

def main():
    # args = parse_args()
    start_training()

if __name__ == "__main__":
    main()

