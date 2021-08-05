import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import math
#smiles = ["N#Cc1cc(F)c(F)c(\\N=N/c2c(C(=O)O)cccc2-c2ccccc2)c1", "COc1cccc(\\N=N/c2ccc(-c3ccccc3)c(C)c2OC)c1C(=O)O","COc1cccc(\\N=N/c2c(C)ccc(-c3ccccc3)c2C(=O)O)c1-c1ccccc1",
#          "N#Cc1cc(C(=O)O)cc(\\N=N/c2cccc(-c3ccccc3)c2C(=O)O)c1C#N", "COc1cc(\\N=N/c2cccc(F)c2)cc(C#N)c1C#N", "COc1cc(C#N)cc(\\N=N/c2cc(-c3ccccc3)cc(C#N)c2C)c1",
#          "COc1c(C(=O)O)ccc(\\N=N/c2ccc(-c3ccccc3)cc2C(=O)O)c1C(=O)O", "COc1cc(\\N=N/c2c(C)cccc2OC)c(C(=O)O)c(-c2ccccc2)c1", "COc1cc(\\N=N/c2ccc(C(=O)O)c(C#N)c2C(=O)O)ccc1C(=O)O",
#          "O=C(O)c1ccc(-c2ccccc2)c(\\N=N/c2ccc(-c3ccccc3)c(C(=O)O)c2C(=O)O)c1"]
#TODO Check if O-O bonds exist that are smaller than 2 smallest O-H distances
smiles = ["COc1c(C(=O)O)ccc(\\N=N/c2ccc(-c3ccccc3)cc2C(=O)O)c1C(=O)O"]
for smile in smiles:
    m = Chem.MolFromSmiles(smile)
    m = Chem.AddHs(m)
    oxy_count = 0
    for atom in m.GetAtoms():
        if atom.GetSymbol() == "O":
            bonded_h = False
            secondary_h = False
            oxy_count+=1
            print("\n")
            print("Oxygen Number: " + str(oxy_count))
            num_bonded_hydrogens = 0
            oxy_index = atom.GetIdx()
            for hydrogen in m.GetAtoms():
                if hydrogen.GetSymbol() == "H":
                    hydro_index = hydrogen.GetIdx()
                    status = AllChem.EmbedMolecule(m)
                    conformer = m.GetConformer()
                    pos = conformer.GetPositions()
                    oxy_pos = pos[oxy_index]
                    hydro_pos = pos[hydro_index]
                    distance = math.sqrt((oxy_pos[0]-hydro_pos[0])**2 + (oxy_pos[1]-hydro_pos[1])**2 +
                                         (oxy_pos[2]-hydro_pos[2])**2)

                    print(distance)
                    if distance < 2 and bonded_h == False:
                        bonded_h = True
                    elif distance < 4 and bonded_h == True:
                        secondary_h = True
                        print("Failed")
                        break




