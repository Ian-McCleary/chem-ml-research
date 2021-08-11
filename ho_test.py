import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import math
from rdkit import RDLogger
smiles = ["COc1cccc(\\N=N/c2c(C)ccc(C#N)c2C)c1C(=O)O", "COc1cccc(C(=O)O)c1\\N=N/c1ccc(C(=O)O)c(C(=O)O)c1C", "Cc1ccc(C(=O)O)c(\\N=N/c2cccc(F)c2C(=O)O)c1F",
"COc1c(C)cccc1\\N=N/c1c(-c2ccccc2)ccc(F)c1C#N", "COc1cccc(\\N=N/c2cc(OC)c(C(=O)O)c(-c3ccccc3)c2)c1OC", "COc1cccc(\\N=N/c2cc(C#N)cc(OC)c2C(=O)O)c1C(=O)O"]
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

def has_covalent_hydrogen_bond(oxygen_index, atom_list, bond_list):
    for i in range(len(bond_list)):
        try:
            connecting_atom = Chem.rdchem.Bond.GetOtherAtomIdx(bond_list[i], oxygen_index)
        except (RuntimeError):
            continue
        if atom_list[connecting_atom].GetSymbol() == "H":
            return True
        else:
            continue
    return False


lg = RDLogger.logger()

lg.setLevel(RDLogger.CRITICAL)
#smiles = ["COc1cccc(\\N=N/c2ccc(-c3ccccc3)c(C)c2OC)c1C(=O)O"]
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
            #print("\n")
            #print("Oxygen Number: " + str(oxy_count))
            has_covalent_bond = has_covalent_hydrogen_bond(i, atom_list, bond_list)
            h_half = False
            bonded_h = False
            bonded_h_val = 0
            for j in range(len(atom_list)):
                b_1 = atom_list[j]
                if b_1.GetSymbol() == "H":
                    #recursively check the side of each hydrogen atom
                    h_half = find_half(bond_list, atom_list, j)
                    #print("answer: ", answer)

                    if not (o_half is True and h_half is True) or not (o_half is False and h_half is False):

                        oxy_pos = pos[i]
                        hydro_pos = pos[j]
                        hydrogen_distance = math.sqrt((oxy_pos[0]-hydro_pos[0])**2 + (oxy_pos[1]-hydro_pos[1])**2 +
                                            (oxy_pos[2]-hydro_pos[2])**2)
                        #print(distance)
                        if hydrogen_distance < 4 and has_covalent_bond is True:
                            failed = False
                            #check for nearby oxygen that could be closer
                            for oxygen in m.GetAtoms():
                                if oxygen.GetSymbol() == "O" and not oxygen.GetIdx() == i:
                                    if not has_covalent_hydrogen_bond(oxygen.GetIdx(), atom_list, bond_list):
                                        oxy2_index = oxygen.GetIdx()
                                        oxy2_pos = pos[oxy2_index]
                                        oxy_distance = math.sqrt(
                                            (hydro_pos[0] - oxy2_pos[0])**2 + (hydro_pos[1] - oxy2_pos[1])**2 +
                                            (hydro_pos[2] - oxy2_pos[2])**2)
                                        #print(oxy_distance)
                                        if oxy_distance < hydrogen_distance:
                                            failed = False
                                            break
                                        else:
                                            failed = True
                                    else:
                                        failed = True
                                if failed == True:
                                    print("Failed")
                                    break
                                else:
                                    print("Passed")



