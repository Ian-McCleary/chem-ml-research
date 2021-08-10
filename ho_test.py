import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import math
from rdkit import RDLogger
#smiles = ["N#Cc1cc(F)c(F)c(\\N=N/c2c(C(=O)O)cccc2-c2ccccc2)c1", "COc1cccc(\\N=N/c2ccc(-c3ccccc3)c(C)c2OC)c1C(=O)O","COc1cccc(\\N=N/c2c(C)ccc(-c3ccccc3)c2C(=O)O)c1-c1ccccc1",
#          "N#Cc1cc(C(=O)O)cc(\\N=N/c2cccc(-c3ccccc3)c2C(=O)O)c1C#N", "COc1cc(\\N=N/c2cccc(F)c2)cc(C#N)c1C#N", "COc1cc(C#N)cc(\\N=N/c2cc(-c3ccccc3)cc(C#N)c2C)c1",
#          "COc1c(C(=O)O)ccc(\\N=N/c2ccc(-c3ccccc3)cc2C(=O)O)c1C(=O)O", "COc1cc(\\N=N/c2c(C)cccc2OC)c(C(=O)O)c(-c2ccccc2)c1", "COc1cc(\\N=N/c2ccc(C(=O)O)c(C#N)c2C(=O)O)ccc1C(=O)O",
#          "O=C(O)c1ccc(-c2ccccc2)c(\\N=N/c2ccc(-c3ccccc3)c(C(=O)O)c2C(=O)O)c1"]
#TODO Implement backtracking instead of recursion. Check each possible route to the nearest N


def find_half(bond_list, atom_list, start):
    visited_list = []
    path = find_nearest_oxygen_or_carbon(start, start, bond_list, atom_list)
    i = path
    while i < len(atom_list):
        a_1 = i
        if i < len(atom_list)-1:
            a_2 = i+1

        if atom_list[a_1].GetSymbol() == "N" and atom_list[a_2].GetSymbol() == "N":
            return False
        i+=1
    return True

def find_nearest_oxygen_or_carbon(current, previous, bond_list, atom_list):
    for i in range(len(bond_list)):
        try:
            connecting_atom = Chem.rdchem.Bond.GetOtherAtomIdx(bond_list[i], current)
        except (RuntimeError):
            continue
        if not connecting_atom == previous:
            if atom_list[connecting_atom].GetSymbol() == "O" or atom_list[connecting_atom].GetSymbol() == "C":
                return connecting_atom
            n = find_nearest_oxygen_or_carbon(connecting_atom, current, bond_list, atom_list)
            if n == -1:
                continue
            elif atom_list[n].GetSymbol() == "O" or atom_list[n].GetSymbol() == "C":
                return n


'''
def find_half(bond_list, atom_list, previous, current):
    for x in range(len(bond_list)):
        try:
            connecting_atom = Chem.rdchem.Bond.GetOtherAtomIdx(bond_list[x], current)
        except (RuntimeError):
            continue
        print(atom_list[connecting_atom].GetSymbol())
        if not connecting_atom == previous:
            if atom_list[connecting_atom].GetSymbol() == "N":
                #print("connecting: ", connecting_atom)
                return connecting_atom
            else:
                next_run = find_half(bond_list, atom_list, current, connecting_atom)
                if atom_list[next_run].GetSymbol() == "N":
                    return next_run
        elif connecting_atom == previous:
            continue
    return -1
    '''





lg = RDLogger.logger()

lg.setLevel(RDLogger.CRITICAL)
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
        #print(atom_list[i].GetSymbol())
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
                    h_half = find_half(bond_list, atom_list, j)
                    #print("answer: ", answer)

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
                    if distance < 1.5 and potential_cov is True:
                        bonded_h = True
                        bonded_h_val = distance
                    elif distance < 4 and potential_cov is False:
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


