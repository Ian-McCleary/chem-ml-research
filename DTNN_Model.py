from rdkit import Chem
import deepchem as dc
import numpy as np

# TODO
# Robust dataset generation from the file structure in cluster.
# Dataset generation should only be done once.

# Meta smiles strings randomly sampled from Batch 4 and 8
m_51677 = 'COc1cc(C#N)c(\\N=N/c2ccccc2C(=O)O)c(C(=O)O)c1'
m_168242 = 'Cc1ccc(-c2ccccc2)c(-c2ccccc2)c1\\N=N/c1ccccc1F'
m_857149 = 'COc1c(C(=O)O)ccc(F)c1\\N=N/c1cc(C#N)cc(C(=O)O)c1'
m_945444 = 'Cc1cc(C(=O)O)ccc1\\N=N/c1cc(C(=O)O)cc(-c2ccccc2)c1'
m_963853 = 'COc1c(\\N=N/c2c(C)cccc2C(=O)O)ccc(F)c1C#N'
m_971449 = 'COc1cc(C(=O)O)cc(\\N=N/c2cccc(F)c2C(=O)O)c1C#N'
m_980818 = 'COc1ccc(\\N=N/c2cc(F)ccc2C(=O)O)c(OC)c1C#N'
m_983269 = 'COc1c(F)cccc1\\N=N/c1c(F)cccc1C(=O)O'
m_998710 = 'Cc1cc(C(=O)O)cc(\\N=N/c2cc(F)cc(C(=O)O)c2)c1C'
m_1004663 = 'N#Cc1ccc(C(=O)O)c(C#N)c1\\N=N/c1cc(C(=O)O)ccc1F'
smile_arr = [m_51677, m_168242, m_857149, m_945444, m_963853, m_971449, m_980818, m_983269, m_998710, m_1004663]

# Energy difference calculated from NEB.out. Is this ground state difference?
# -metastable - (-stable)
y_51677 = [-1501.474570+1501.463127]
y_168242 = [-1604.911934+1604.999148]
y_857149 = [-1610.012195+1610.101441]
y_945444 = [-1630.715327+1630.774141]
y_963853 = [-1449.851656+1449.938197]
y_971449 = [-1610.196049+1610.266051]
y_980818 = [-1539.706923+1539.912726]
y_983269 = [-1389.671748+1389.934824]
y_998710 = [-1486.485683+1486.563960]
y_1004663 = [-1554.285506+1554.414150]
output_y = np.array([y_51677, y_168242, y_857149, y_945444, y_963853, y_971449, y_980818, y_983269, y_998710, y_1004663])

input_X = np.zeros([10, 45, 45])
id_arr = np.array([51677, 168242, 857149, 945444, 963853, 971449, 980818, 983269, 998710, 1004663])

# Generate Coulomb matrix with 2 conforms. Should be a 2d vector
for i in range(len(smile_arr)):
    generator = dc.utils.ConformerGenerator(max_conformers=1)
    azo_mol = generator.generate_conformers(Chem.MolFromSmiles(smile_arr[i]))
    coulomb_mat = dc.feat.CoulombMatrix(max_atoms=45, remove_hydrogens=True)
    features = coulomb_mat(azo_mol)
    input_X[i] = features

data = dc.data.NumpyDataset(X=input_X, y=output_y, ids=id_arr, n_tasks=1)
# Splits dataset into train/validation/test
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset=data)
print(len(train_dataset))
print(output_y)

metric = [
    dc.metrics.Metric(dc.metrics.mean_absolute_error, mode="regression"),
    dc.metrics.Metric(dc.metrics.mean_squared_error, mode="regression")
]

model = dc.models.DTNNModel(
    n_tasks=1,
    n_embedding=10,
    # n_hidden=45,
    mode="regression",
    dropout=0.1,
    learning_rate=0.001
)
model.fit(train_dataset)
# How well the model fit's the training subset of our data
train_scores = model.evaluate(train_dataset, metric)
# Validation of the model over several training iterations.
valid_score = model.evaluate(valid_dataset, metric)
# How well the model generalizes the rest of the data
test_score = model.evaluate(test_dataset, metric)
print("Training Scores: ")
print(train_scores)
print("Validity Scores: ")
print(valid_score)
print("Test Scores: ")
print(test_score)

#generated_batch = dc.default_generator(data, epochs=2, mode='fit', deterministic=False, pad_batches=True)
#print(generated_batch)

