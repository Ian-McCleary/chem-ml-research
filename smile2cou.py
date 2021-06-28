from rdkit import Chem
import deepchem as dc

smile1 = "COc1cc(C#N)c(\\N=N/c2ccccc2C(=O)O)c(C(=O)O)c1"

generator = dc.utils.ConformerGenerator(max_conformers=5)
azo_mol = generator.generate_conformers(Chem.MolFromSmiles(smile1))
print("Number of available conformers for propane: ", len(azo_mol.GetConformers()))

coulomb_mat = dc.feat.CoulombMatrix(max_atoms=20)
features = coulomb_mat(azo_mol)
print(features)