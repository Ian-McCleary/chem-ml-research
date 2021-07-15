from rdkit import Chem
import deepchem as dc
import numpy as np
import pandas as pd

loader = dc.data.CSVLoader(["task1"], feature_field="smiles", id_field="ids", featurizer=dc.feat.CircularFingerprint(size=2048, radius=2))
data = loader.create_dataset("Datasets/dataset_100.csv")

y_vals = data.y

transformer = dc.trans.NormalizationTransformer(dataset=data, transform_y=True)
dataset = transformer.transform(data)

transformed_vals = dataset.y

file_name = "transformer_visualization.csv"
df = pd.DataFrame(list(zip(y_vals, transformed_vals)), columns=["train_losses", "valid_losses"])
df.to_csv(file_name)