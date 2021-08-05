import numpy as np
from rdkit import Chem

smile = "COc1cc(C#N)c(\\N=N/c2ccccc2C(=O)O)c(C(=O)O)c1"

m = Chem.MolFromSmiles(smile)
for atom in m.GetAtoms():
    if atom.GetSymbol() == "O":
        if atom.GetTotalNumHs() > 1:
            print(atom.GetSymbol(), atom.GetExplicitValence(), atom.GetTotalNumHs())