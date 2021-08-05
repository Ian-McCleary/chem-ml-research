import numpy as np
from rdkit import Chem

smile = "COc1cc(C#N)c(\\N=N/c2ccccc2C(=O)O)c(C(=O)O)c1"

m = Chem.MolFromSmiles(smile)
atoms = m.GetAtoms()
for i in range(len(atoms)):
    if atoms[i] == "O":
        print("O")
    print(atoms[i].getSymbol())