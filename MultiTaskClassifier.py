from random import randrange
import pandas as pd
import numpy as np

from ghost import optimize_threshold_from_predictions

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
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
    #metrics = [dc.metrics.Metric(dc.metrics.roc_auc_score)]

    #model = mtc_fixed_param_model(task_count=task_count, n_features=n_features)
    
    model = mtc_hyperparameter_optimization(train_dataset, valid_dataset, metric)
    #threshold_training(model, train_dataset, valid_dataset, test_dataset, metric)
    all_loss = loss_over_epoch(model, train_dataset, valid_dataset, test_dataset, metric, 50)
    #k_fold_validation(model, data)
    # hyperparameter_optimization()
    #file_name = "mtc_10k_test.csv"
    #df = pd.DataFrame(list(zip(all_loss[0], all_loss[1], all_loss[2], all_loss[3])), columns=[
    #        "train roc_auc","train f1", "valid roc_auc", "valid f1"])

    #df.to_csv(file_name)

def mtc_hyperparameter_optimization(train_dataset, valid_dataset, metric):
    task_count = len(train_dataset.y[0])
    n_features = len(train_dataset.X[0])
    l_rate_scheduler = dc.models.optimizers.ExponentialDecay(0.0002, 0.9, 15)
    params_dict = {
        'n_tasks': [task_count],
        'n_features': [n_features],
        'layer_sizes': [[256, 512, 1024], [512, 1024, 2048], [1024, 1024, 2048], [1024, 512, 2048]],
        'dropouts': [0.2, 0.5, 0.4, 0.6],
        'n_classes': [2],
        'weight_decay_penalty_type': ["l1","l2"],
        'weight_decay_penalty': [0.002, 0.0002, 0.00002],
        'learning_rate': [0.0001, 0.00001],
        'mode': ["classification"]
    }

    optimizer = dc.hyper.GridHyperparamOpt(dc.models.MultitaskClassifier)
    best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
        params_dict, train_dataset, valid_dataset, metric)
    print(best_hyperparams)
    return best_model


# arbitrary fixed parameter model for quick testing
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


# Create the binary classification vector based off a given threshold.
def get_classification(predict, threshold):
    classification = []
    for i in range(len(predict)):
        #print(predict[i][0][0])
        if predict[i][0][0] > threshold:
            classification.append(0)
        else:
            classification.append(1)
    return classification


def threshold_training(model, train_dataset, valid_dataset, test_dataset, metric):
    callback = dc.models.ValidationCallback(valid_dataset, 10, metric)
    model.fit(train_dataset, nb_epoch=5, callbacks=callback)

    training_pred = model.predict(train_dataset)
    thresh_list = []
    x = 0.50
    while x <= 0.95:
        thresh_list.append(x)
    

    #threshold = optimize_threshold_from_predictions(labels = train_dataset.y, probs = training_pred[:,1], thresholds = thresh_list,ThOpt_metrics= 'Kappa', N_subsets=100, subsets_size=0.2)
    threshold = find_threshold(model, train_dataset, valid_dataset)
    print("Threshold: ", threshold)
    test_pred = model.predict(test_dataset)
    test_classification = get_classification(test_pred, threshold)
    #test_classification = dc.metrics.handle_classification_mode(test_pred, classification_handling_mode='threshold')
    test_f1 = f1_score(test_dataset.y, test_classification, average='binary', pos_label=1)
    print("Test f1_score:", test_f1)
    test_recall = recall_score(test_dataset.y, test_classification, average='binary', pos_label=1)
    print("Test Recall:", test_recall)

# Calculate loss or other performance metrics over training epochs
def loss_over_epoch(model, train_dataset, valid_dataset, test_dataset, metric, epochs):
    #metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
    # Threshold to use for classification predictions
    model_copy = model
    #threshold = find_threshold(model_copy, train_dataset, valid_dataset)
    
    train_classification = []
    train_m1 = []
    train_m2 = []
    train_riso = []
    train_vert = []

    valid_classification = []
    valid_m1 = []
    valid_m2 = []
    valid_riso = []
    valid_vert = []
    all_loss = []
    for i in range(epochs):
        loss = model.fit(train_dataset, nb_epoch=1)
        train = model.evaluate(train_dataset, metric, per_task_metrics=True)
        valid = model.evaluate(valid_dataset, metric, per_task_metrics=True)
        train_pred = model.predict(train_dataset)
        valid_pred = model.predict(valid_dataset)
        # print(valid[0]["mean_absolute_error"])
        # print(valid[1]["mean_absolute_error"])
        # print(valid[1]["mean_absolute_error"][0])
        #print(train_pred)

        '''
        train_classification = get_classification(train_pred, threshold)
        valid_classification = get_classification(valid_pred, threshold)

        #train_classification = dc.metrics.handle_classification_mode(train_pred, classification_handling_mode='threshold')
        #valid_classification = dc.metrics.handle_classification_mode(valid_pred, classification_handling_mode='threshold')
        #print(train_classification)
        train_f1 = f1_score(train_dataset.y, train_classification, average='binary', pos_label=1)
        train_recall = recall_score(train_dataset.y, train_classification, average='binary', pos_label=1)
        train_m1.append(train_f1)
        train_m2.append(train[0]["roc_auc_score"])

        valid_f1 = f1_score(valid_dataset.y, valid_classification, average='binary')
        valid_recall = recall_score(valid_dataset.y, valid_classification, average='binary', pos_label=1)
        valid_m1.append(valid_f1)
        valid_m2.append(valid[0]["roc_auc_score"])
        


    all_loss.append(train_m1)
    all_loss.append(train_m2)

    all_loss.append(valid_m1)
    all_loss.append(valid_m2)
    '''
    thresh_list = []
    x = 0.50
    while x <= 0.95:
        thresh_list.append(x)
    threshold = optimize_threshold_from_predictions(labels = train_dataset.y, probs = train_pred[:,1], thresholds = thresh_list,ThOpt_metrics= 'Kappa', N_subsets=100, subsets_size=0.2)
    print("Threshold: ", threshold)
    train_classification = get_classification(train_pred, threshold)
    train_scores = model.evaluate(train_dataset, metric, per_task_metrics=True)
    print("Train Average_precision:")
    print(train_scores[0]["roc_auc_score"])
    train_f1 = f1_score(train_dataset.y, train_classification, average='binary', pos_label=1)

    print("Train f1_score:", train_f1)
    train_recall = recall_score(train_dataset.y, train_classification, average='binary', pos_label=1)
    print("Train Recall:", train_recall)

    test_scores = model.evaluate(test_dataset, metric, per_task_metrics=True)
    print("Test Average_precision:")
    print(test_scores[0]["roc_auc_score"])
    
    test_pred = model.predict(test_dataset)
    test_classification = get_classification(test_pred, threshold)
    #test_classification = dc.metrics.handle_classification_mode(test_pred, classification_handling_mode='threshold')
    test_f1 = f1_score(test_dataset.y, test_classification, average='binary', pos_label=1)
    print("Test f1_score:", test_f1)
    test_recall = recall_score(test_dataset.y, test_classification, average='binary', pos_label=1)
    print("Test Recall:", test_recall)
    return all_loss

# Find the threshold for classification predictions from 0.5 to 0.9
# Uses the hyperparam model
def find_threshold(model_origional, train_dataset, valid_dataset):
    task_count = len(train_dataset.y[0])
    n_features = len(train_dataset.X[0])
    thresh_list = []
    max_f1 = 0
    threshold = 0.5
    while threshold < 0.9:
        model = model_origional
        model.fit(train_dataset, nb_epoch=20)
        valid_pred = model.predict(valid_dataset)
        valid_classification = get_classification(valid_pred, threshold)
        f1 = f1_score(valid_dataset.y, valid_classification, average='binary')
        thresh_list.append([threshold,f1])
        threshold += 0.05
    max = 0
    idx = 0
    for i in range(len(thresh_list)):
        if thresh_list[i][1] > max:
            max = thresh_list[i][1]
            idx = i
    
    return thresh_list[idx][0]
    


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

