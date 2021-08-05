import numpy as np
from rdkit import Chem

smiles = ["COc1cc(C#N)c(\\N=N/c2ccccc2C(=O)O)c(C(=O)O)c1", "COc1c(\\N=N/c2c(C)cccc2C(=O)O)ccc(F)c1C#N", "Cc1cccc(\\N=N/c2c(C(=O)O)ccc(-c3ccccc3)c2C(=O)O)c1C(=O)O",
          "N#Cc1cccc(C(=O)O)c1\\N=N/c1cc(C(=O)O)cc(-c2ccccc2)c1C(=O)O"]
for smile in smiles:
    m = Chem.MolFromSmiles(smile)
    for atom in m.GetAtoms():
        if atom.GetSymbol() == "O":
            if atom.GetTotalNumHs() > 1:
                print(smile, atom.GetSymbol(), atom.GetExplicitValence(), atom.GetTotalNumHs())