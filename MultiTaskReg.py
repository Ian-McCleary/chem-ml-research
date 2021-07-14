from rdkit import Chem
import deepchem as dc
import numpy as np
import pandas as pd

loader = dc.data.CSVLoader(["task1"], feature_field="smiles", id_field="ids", featurizer=dc.feat.CircularFingerprint(size=2048, radius=2))
data = loader.create_dataset("Datasets/dataset_1000.csv")
transformer = dc.trans.NormalizationTransformer(dataset=data, transform_y=True)
dataset = transformer.transform(data)

# Splits dataset into train/validation/test
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=dataset)
task_count = len(dataset.y[0])
n_features = len(dataset.X[0])

print(task_count)
#tasks, datasets, transformers = dc.molnet.load_hiv(featurizer='ECFP', split='scaffold')
#train_dataset, valid_dataset, test_dataset = datasets

params_dict = {
    'n_tasks': [task_count],
    'n_features': [n_features],
    'layer_sizes': [[500], [1000]],
    'dropouts': [0.2, 0.5],
    'learning_rate': [0.001, 0.0001]
}
#print(data.y)


optimizer = dc.hyper.GridHyperparamOpt(dc.models.MultitaskRegressor)
metric = dc.metrics.Metric(dc.metrics.rms_score)
best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
        params_dict, train_dataset, valid_dataset, metric, [transformer])
print(all_results)

model = dc.models.MultitaskRegressor(
    n_tasks=task_count,
    n_features=n_features,
    layer_sizes=best_hyperparams[2],
    dropouts=best_hyperparams[3],
    learning_rate=best_hyperparams[4],
    mode="regression"
  )


# Fit trained model
# test
train_losses = []
for i in range(100):
  loss = model.fit(train_dataset, nb_epoch=1)
  print("loss: %s" % str(loss))
  train_losses.append(loss)
print("losses")
print(train_losses)
print("\n")
print("Valid_dataset losses:")

valid_losses = []
for i in range(100):
    loss = model.fit(valid_dataset, nb_epoch=1)
    print("loss: %s" % str(loss))
    valid_losses.append(loss)
print("losses")
print(valid_losses)

df = pd.DataFrame(list(zip(train_losses, valid_losses)), columns=["train_losses", "valid_losses"])
df.to_csv("mtr_losses.csv")