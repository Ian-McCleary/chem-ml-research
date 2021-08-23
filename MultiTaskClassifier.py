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
    loader = dc.data.CSVLoader(["binary_negativity"], feature_field="smiles", id_field="ids",
                                featurizer=dc.feat.CircularFingerprint(size=2048, radius=2))
    data = loader.create_dataset("Datasets/dataset_4task_10k.csv")
    

    # Splits dataset into train/validation/test
    splitter = dc.splits.RandomSplitter()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset=data, frac_train=0.70, frac_valid=0.15, frac_test=0.15, seed=dataseed)
    task_count = len(data.y[0])
    n_features = len(data.X[0])

    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
    #metrics = [dc.metrics.Metric(dc.metrics.auc), dc.metrics.Metric(dc.metrics.precision_score), dc.metrics.Metric(dc.metrics.roc_auc_score)]

    #model = mtc_fixed_param_model(task_count=task_count, n_features=n_features)
    model = mtc_hyperparameter_optimization(train_dataset, valid_dataset, metric)
    all_loss = loss_over_epoch(model, train_dataset, valid_dataset, test_dataset, metric, 250)
    #k_fold_validation(model, data)
    # hyperparameter_optimization()
    file_name = "mtc_10k_test.csv"
    df = pd.DataFrame(list(zip(all_loss[0], all_loss[1])), columns=[
            "train roc_auc", "valid roc_auc"])

    df.to_csv(file_name)

def mtc_hyperparameter_optimization(train_dataset, valid_dataset, metric):
    task_count = len(train_dataset.y[0])
    n_features = len(train_dataset.X[0])
    l_rate_scheduler = dc.models.optimizers.ExponentialDecay(0.0002, 0.9, 15)
    params_dict = {
        'n_tasks': [task_count],
        'n_features': [n_features],
        'layer_sizes': [[256, 512, 1024], [128, 256, 512], [64, 128, 256]],
        'dropouts': [0.2, 0.5, 0.4,],
        'n_classes': [2],
        'learning_rate': [0.0001, 0.00001],
        'mode': ["classification"]
    }

    optimizer = dc.hyper.GridHyperparamOpt(dc.models.MultitaskClassifier)
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
        params_dict, train_dataset, valid_dataset, metric)
    print(best_hyperparams)
    return best_model



def mtc_fixed_param_model(task_count, n_features):
    model = dc.models.MultitaskClassifier(
        n_tasks=task_count,
        n_features=n_features,
        layer_sizes=[32, 64, 128],
        weight_decay_penalty_type="l2",
        dropouts=0.3,
        learning_rate=0.0001,
        n_classes=2,
        mode="classification"
    )
    return model

def loss_over_epoch(model, train_dataset, valid_dataset, test_dataset, metric, epochs):
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
    for i in range(epochs):
        loss = model.fit(train_dataset, nb_epoch=1)
        train = model.evaluate(train_dataset, metric, per_task_metrics=True)
        valid = model.evaluate(valid_dataset, metric, per_task_metrics=True)
        # print(valid[0]["mean_absolute_error"])
        # print(valid[1]["mean_absolute_error"])
        # print(valid[1]["mean_absolute_error"][0])
        #train_mean.append(train[0]["precision_score"])
        train_eiso.append(train[0]["roc_auc_score"])


        #valid_mean.append(valid[0]["precision_score"])
        valid_eiso.append(valid[0]["roc_auc_score"])

    # all_loss.extend([train_mean, train_eiso, train_riso, train_vert])mean
    # all_loss.extend([valid_mean, valid_eiso, valid_riso, valid_vert])
    #all_loss.append(train_mean)
    all_loss.append(train_eiso)

    #all_loss.append(valid_mean)
    all_loss.append(valid_eiso)

    #[transformer]
    test_scores = model.evaluate(test_dataset, metric, per_task_metrics=True)
    #print("Test roc_auc:")
    #print(test_scores[0]["auc"])
    #print("Test roc_curve:")
    #print(test_scores[0]["precision_score"])
    print("Test Average_precision:")
    print(test_scores[0]["roc_auc_score"])
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


def main():
    # args = parse_args()
    start_training()

if __name__ == "__main__":
    main()

