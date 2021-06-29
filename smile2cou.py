from rdkit import Chem
import deepchem as dc
import numpy as np

#TODO
#Robust dataset generation from the file structure in cluster.
#Dataset generation should only be done once.

#Meta smiles strings randomly sampled from Batch 4 and 8
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
smile_arr = [m_51677,m_168242,m_857149,m_945444,m_963853,m_971449,m_980818,m_983269,m_998710,m_1004663]

input_X = np.zeros([10,1,45,45])

#Generate Coulomb matrix with 2 conforms. Should be a 2d vector
for i in range(len(smile_arr)):
    generator = dc.utils.ConformerGenerator(max_conformers=1)
    azo_mol = generator.generate_conformers(Chem.MolFromSmiles(smile_arr[i]))
    coulomb_mat = dc.feat.CoulombMatrix(max_atoms=45)
    features = coulomb_mat(azo_mol)
    input_X[i] = features

print("TESTING COULOMB DATASET INPUT \n")
for x in range(len(input_X)):
    print(input_X[x])
    print("\n")

#Test set from Deepchem
tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = datasets

print(test_dataset)
print("\n")
print(test_dataset.y)
print("\n")
print(test_dataset.X)




#Generate Coulomb matrix
#generator = dc.utils.ConformerGenerator(max_conformers=5)
#azo_mol = generator.generate_conformers(Chem.MolFromSmiles(smile1))
#print("Number of available conformers for propane: ", len(azo_mol.GetConformers()))
#coulomb_mat = dc.feat.CoulombMatrix(max_atoms=50)
#features = coulomb_mat(azo_mol)
#print(features)

