import numpy as np
from rdkit import Chem
import rdkit
smiles = ["N#Cc1cc(F)c(F)c(\\N=N/c2c(C(=O)O)cccc2-c2ccccc2)c1", "COc1cccc(\\N=N/c2ccc(-c3ccccc3)c(C)c2OC)c1C(=O)O","COc1cccc(\\N=N/c2c(C)ccc(-c3ccccc3)c2C(=O)O)c1-c1ccccc1",
          "N#Cc1cc(C(=O)O)cc(\\N=N/c2cccc(-c3ccccc3)c2C(=O)O)c1C#N", "COc1cc(\\N=N/c2cccc(F)c2)cc(C#N)c1C#N", "COc1cc(C#N)cc(\\N=N/c2cc(-c3ccccc3)cc(C#N)c2C)c1",
          "COc1c(C(=O)O)ccc(\\N=N/c2ccc(-c3ccccc3)cc2C(=O)O)c1C(=O)O", "COc1cc(\\N=N/c2c(C)cccc2OC)c(C(=O)O)c(-c2ccccc2)c1", "COc1cc(\\N=N/c2ccc(C(=O)O)c(C#N)c2C(=O)O)ccc1C(=O)O",
          "O=C(O)c1ccc(-c2ccccc2)c(\\N=N/c2ccc(-c3ccccc3)c(C(=O)O)c2C(=O)O)c1"]
for smile in smiles:
    m = Chem.MolFromSmiles(smile)
    m = Chem.AddHs(m)
    for atom in m.GetAtoms():
        if atom.GetSymbol() == "O":
            for hydrogen in m.GetAtoms():
                if hydrogen.GetSymbol() == "H":
                    status = AllChem.EmbedMolecule(m)
                    conformer = m.GetConformer()
                    oxy_pos = m.GetAtomPosition(conformer, atom.getIdx())
                    print(oxy_pos)



