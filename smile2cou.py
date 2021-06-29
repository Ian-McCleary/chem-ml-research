from rdkit import Chem
import deepchem as dc

#Meta smiles strings randomly sampled from Batch 4
smile1 = 'COc1cc(C#N)c(\\N=N/c2ccccc2C(=O)O)c(C(=O)O)c1'
smile2 = 'Cc1ccc(-c2ccccc2)c(-c2ccccc2)c1\\N=N/c1ccccc1F'
smile3 = 'Cc1cc(C(=O)O)cc(\\N=N/c2cc(F)cc(C(=O)O)c2)c1C'
smile4 = 'N#Cc1ccc(C(=O)O)c(C#N)c1\\N=N/c1cc(C(=O)O)ccc1F'


tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = datasets

print(test_dataset)
print("\n")
print(test_dataset.y)
print("\n")
print(test_dataset.x)

#Generate Coulomb matrix
#generator = dc.utils.ConformerGenerator(max_conformers=5)
#azo_mol = generator.generate_conformers(Chem.MolFromSmiles(smile1))
#print("Number of available conformers for propane: ", len(azo_mol.GetConformers()))
#coulomb_mat = dc.feat.CoulombMatrix(max_atoms=50)
#features = coulomb_mat(azo_mol)
#print(features)

