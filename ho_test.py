import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import math
#smiles = ["N#Cc1cc(F)c(F)c(\\N=N/c2c(C(=O)O)cccc2-c2ccccc2)c1", "COc1cccc(\\N=N/c2ccc(-c3ccccc3)c(C)c2OC)c1C(=O)O","COc1cccc(\\N=N/c2c(C)ccc(-c3ccccc3)c2C(=O)O)c1-c1ccccc1",
#          "N#Cc1cc(C(=O)O)cc(\\N=N/c2cccc(-c3ccccc3)c2C(=O)O)c1C#N", "COc1cc(\\N=N/c2cccc(F)c2)cc(C#N)c1C#N", "COc1cc(C#N)cc(\\N=N/c2cc(-c3ccccc3)cc(C#N)c2C)c1",
#          "COc1c(C(=O)O)ccc(\\N=N/c2ccc(-c3ccccc3)cc2C(=O)O)c1C(=O)O", "COc1cc(\\N=N/c2c(C)cccc2OC)c(C(=O)O)c(-c2ccccc2)c1", "COc1cc(\\N=N/c2ccc(C(=O)O)c(C#N)c2C(=O)O)ccc1C(=O)O",
#          "O=C(O)c1ccc(-c2ccccc2)c(\\N=N/c2ccc(-c3ccccc3)c(C(=O)O)c2C(=O)O)c1"]
#TODO Check if O-O bonds exist that are smaller than 2 smallest O-H distances
# Is it possible to check if atom index is the same in position array?
# We may need to check that that O-H bonding is not happening on opposite sides of the N=N bond.

def find_half(bond_list, atom_list, index):
    for x in range(len(bond_list)):
        try:
            connecting_atom = Chem.rdchem.Bond.GetOtherAtomIdx(bond_list[x], index)
            print("connecting: ",connecting_atom)
        except (RuntimeError):
            pass
            continue
        if connecting_atom in range(len(atom_list)):
            if atom_list[connecting_atom] == "N":
                return connecting_atom
            else:
                return find_half(bond_list, atom_list, index)


smiles = ["COc1cccc(\\N=N/c2ccc(-c3ccccc3)c(C)c2OC)c1C(=O)O"]
for smile in smiles:
    print(smile)
    m = Chem.MolFromSmiles(smile)
    m = Chem.AddHs(m)
    #add_h_smile = Chem.rdmolfiles.MolToSmiles(mol=m, allHsExplicit=True)
    #print(add_h_smile)
    #m1 = Chem.MolFromSmiles(add_h_smile)
    status = AllChem.EmbedMolecule(m)
    conformer = m.GetConformer()
    pos = conformer.GetPositions()
    oxy_count = 0
    # false = first half, true = second half
    o_half = False
    atom_list = m.GetAtoms()
    bond_list = m.GetBonds()
    for i in range(len(atom_list)):
        #check which half oxygen is on
        print(atom_list[i].GetSymbol())
        a_1 = atom_list[i]
        if i < len(atom_list)-1:
            a_2 = atom_list[i+1]
        if a_1.GetSymbol() == "N" and a_2.GetSymbol() == "N":
            o_half = True
        if a_1.GetSymbol() == "O":

            oxy_count+=1
            print("\n")
            print("Oxygen Number: " + str(oxy_count))
            num_bonded_hydrogens = 0
            h_half = False
            bonded_h = False
            bonded_h_val = 0
            for j in range(len(atom_list)):
                b_1 = atom_list[j]
                if b_1.GetSymbol() == "H":
                    #recursively check the side of each hydrogen atom
                    answer = find_half(bond_list, atom_list, j)
                    print("answer: ", answer)
                    if (o_half is True and h_half is True) or (o_half is False and h_half is False):
                        potential_cov = True
                    else:
                        potential_cov = False

                    print("o_half: ",o_half,"  h_half: ",h_half)

                    oxy_pos = pos[i]
                    hydro_pos = pos[j]
                    distance = math.sqrt((oxy_pos[0]-hydro_pos[0])**2 + (oxy_pos[1]-hydro_pos[1])**2 +
                                         (oxy_pos[2]-hydro_pos[2])**2)
                    print(distance)
                    if distance < 1.5 and potential_cov == True:
                        bonded_h = True
                        bonded_h_val = distance
                    elif distance < 4 and potential_cov == False:
                        failed = False
                        #check for nearby oxygen that could be closer
                        for oxygen in m.GetAtoms():
                            if oxygen.GetSymbol() == "O" and not oxygen.GetIdx() == i:
                                oxy2_index = oxygen.GetIdx()
                                oxy2_pos = pos[oxy2_index]
                                oxy_distance = math.sqrt(
                                    (oxy_pos[0] - oxy2_pos[0])**2 + (oxy_pos[1] - oxy2_pos[1])**2 +
                                    (oxy_pos[2] - oxy2_pos[2])**2)
                                #print(oxy_distance)
                                if oxy_distance < bonded_h_val or oxy_distance < distance:
                                    failed = False
                                    break
                                else:
                                    failed = True
                        if failed == True:
                            print("Failed")
                        else:
                            print("Passed")


