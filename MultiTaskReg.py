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
    data = loader.create_dataset("Datasets/dataset_10k_3task.csv")
    
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

    #model = fixed_param_model(task_count=task_count, n_features=n_features)
    model = hyperparameter_optimization(train_dataset, valid_dataset, transformer, metric)
    all_loss = train_loss(model, train_dataset, valid_dataset, test_dataset, metrics, [transformer])
    # hyperparameter_optimization()
    file_name = "mtr_10k_hyper.csv"
    df = pd.DataFrame(list(zip(all_loss[0], all_loss[1], all_loss[2], all_loss[3], all_loss[4], all_loss[5], all_loss[6], all_loss[7])), columns=[
        "train_mean", "train_eiso", "train_riso", "train_vert", "valid_mean", "valid_eiso", "valid_riso", "valid_vert"])

    df.to_csv(file_name)



def k_fold_validation(model):
    eiso_scores = []
    riso_scores = []
    vert_scores = []

    loader = dc.data.CSVLoader(["task1", "task2", "task3"], feature_field="smiles", id_field="ids",
                               featurizer=dc.feat.CircularFingerprint(size=2048, radius=2))
    data = loader.create_dataset("Datasets/dataset_3task_10k.csv")

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
        test_score = model.evaluate(test_dataset, metric, per_task_metrics=True)
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
def train_loss(model, train_dataset, valid_dataset, test_dataset, metric, transformer):
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

        train_mean.append(train[0]["rms_score"])
        train_eiso.append(train[1]["rms_score"][0])
        train_riso.append(train[1]["rms_score"][1])
        train_vert.append(train[1]["rms_score"][2])

        valid_mean.append(valid[0]["rms_score"])
        valid_eiso.append(valid[1]["rms_score"][0])
        valid_riso.append(valid[1]["rms_score"][1])
        valid_vert.append(valid[1]["rms_score"][2])
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


def main():
    # args = parse_args()
    start_training()

if __name__ == "__main__":
    main()

