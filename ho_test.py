import numpy as np
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
import math
from rdkit import RDLogger
import csv
smiles = ["COc1cccc(\\N=N/c2c(C)ccc(C#N)c2C)c1C(=O)O", "COc1cccc(C(=O)O)c1\\N=N/c1ccc(C(=O)O)c(C(=O)O)c1C", "Cc1ccc(C(=O)O)c(\\N=N/c2cccc(F)c2C(=O)O)c1F",
"COc1c(C)cccc1\\N=N/c1c(-c2ccccc2)ccc(F)c1C#N", "COc1cccc(\\N=N/c2cc(OC)c(C(=O)O)c(-c3ccccc3)c2)c1OC", "COc1cccc(\\N=N/c2cc(C#N)cc(OC)c2C(=O)O)c1C(=O)O"]

passed_smiles = ["COc1cc(\\N=N/c2c(C)ccc(-c3ccccc3)c2C#N)ccc1C(=O)O", "Cc1ccc(\\N=N/c2c(F)cc(C(=O)O)cc2C(=O)O)c(C(=O)O)c1", "COc1cc(C#N)ccc1\\N=N/c1cc(C)c(C)cc1-c1ccccc1",
                 "N#Cc1ccccc1\\N=N/c1cc(C(=O)O)ccc1-c1ccccc1", "O=C(O)c1cc(F)c(\\N=N/c2cccc(F)c2)c(C(=O)O)c1", "COc1ccc(OC)c(\\N=N/c2c(F)ccc(-c3ccccc3)c2C#N)c1"]

#negative_smiles = ["COc1cc(C(=O)O)cc(\\N=N/c2c(F)cccc2C(=O)O)c1OC"]

# Find the nearest oxygen or carbon for a given hydrogen
# This makes finding half much quicker for hydrogens
def find_nearest_oxygen_or_carbon(atom_list, bond_list, start, tracking_list):
    tracking_list.append(start)
    for i in range(len(bond_list)):
        try:
            connecting_atom = Chem.rdchem.Bond.GetOtherAtomIdx(bond_list[i], start)
        except (RuntimeError):
            continue
        if connecting_atom not in tracking_list:
            if atom_list[connecting_atom].GetSymbol() == "O" or atom_list[connecting_atom].GetSymbol() == "C":
                return connecting_atom
            n = find_nearest_oxygen_or_carbon(atom_list, bond_list, connecting_atom, tracking_list)
            if atom_list[n].GetSymbol() == "O" or atom_list[n].GetSymbol() == "C":
                return n

# Return true if an oxygen has a direct bond with a hydrogen
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


# Test all possible bond paths to the N=N bond.
# Return false for "first" half, True for "second" half.
def backtracking_find_half(atom_list, bond_list, start, tracking_list):
    tracking_list.append(start)
    has_connecting_n = False
    #print(atom_list[start].GetSymbol())
    for x in range(len(bond_list)):
        try:
            connecting_n = Chem.rdchem.Bond.GetOtherAtomIdx(bond_list[x], start)
            bond_type = bond_list[x].GetBondTypeAsDouble()
        except (RuntimeError):
            continue
        if atom_list[connecting_n].GetSymbol() == "N":
            has_connecting_n = True
    if atom_list[start].GetSymbol() == "N" and has_connecting_n is True:
        return start
    else:
        for y in range(len(bond_list)):
            try:
                connecting_atom = Chem.rdchem.Bond.GetOtherAtomIdx(bond_list[y], start)
            except (RuntimeError):
                continue
            if connecting_atom not in tracking_list:
                return_val = backtracking_find_half(atom_list, bond_list, connecting_atom, tracking_list)
                if not return_val == -1:
                    return return_val
        return -1

# false = first half, true = second half
def find_half(atom_list, bond_list, start, tracking_list):
    n_pos = backtracking_find_half(atom_list, bond_list, start, tracking_list)
    if atom_list[n_pos].GetSymbol() == "N" and atom_list[n_pos+1].GetSymbol() == "N":
        return False
    elif atom_list[n_pos].GetSymbol() == "N" and atom_list[n_pos-1].GetSymbol() == "N":
        return True

# Get the distance of the nearest oxygen in case of carboxylate
def nearest_oxygen_distance(atom_list, pos_array, oxy_pos):
    min_distance = 1000
    for x in range(len(atom_list)):
        if atom_list[x].GetSymbol() == "O":
            oxy2_pos = pos_array[x]
            oxygen_distance = math.sqrt((oxy_pos[0] - oxy2_pos[0]) ** 2 + (oxy_pos[1] - oxy2_pos[1]) ** 2 +
                                          (oxy_pos[2] - oxy2_pos[2]) ** 2)
            if oxygen_distance < min_distance:
                min_distance = oxygen_distance
    return min_distance

def get_hydrogen_index(oxygen_idx, atom_list, bond_list):
    for x in range(len(bond_list)):
        try:
            connecting_n = Chem.rdchem.Bond.GetOtherAtomIdx(bond_list[x], oxygen_idx)
        except (RuntimeError):
            continue
        if atom_list[connecting_n].GetSymbol() == "H":
            return connecting_n
        

# Driver method
def has_hydrogen_bond(smile, cutoff):
    lg = RDLogger.logger()

    lg.setLevel(RDLogger.CRITICAL)

    #print(smile)
    m = Chem.MolFromSmiles(smile)
    m = Chem.AddHs(m)
    status = AllChem.EmbedMolecule(m, randomSeed=123)
    conformer = m.GetConformer()
    pos = conformer.GetPositions()
    oxy_count = 0

    atom_list = m.GetAtoms()
    bond_list = m.GetBonds()
    for i in range(len(atom_list)):
        # check which half oxygen is on
        # print(atom_list[i].GetSymbol())
        if atom_list[i].GetSymbol() == "O":
            track = []
            o_half = find_half(atom_list, bond_list, i, track)
            tracking_list = []
            oxy_count += 1
            # print("\n")
            # print("Oxygen Number: " + str(oxy_count))
            has_covalent_bond = has_covalent_hydrogen_bond(i, atom_list, bond_list)
            if has_covalent_bond == True:
                hydrogen_idx = get_hydrogen_index(i, atom_list, bond_list)
                hydrogen_pos=pos[hydrogen_idx]
                for j in range(len(atom_list)):
                    b_1 = atom_list[j]
                    if b_1.GetSymbol() == "O":

                        track_list = []
                        o_half2 = find_half(atom_list, bond_list, j, track_list)
                        # print("answer: ", answer)
                        # print(o_half, h_half)
                        if (o_half is True and o_half2 is False) or (o_half is False and o_half2 is True):
                            failed = False
                            oxy_pos = pos[i]
                            opposite_oxy_pos = pos[j]
                            hydrogen_distance = math.sqrt(
                                (hydrogen_pos[0] - opposite_oxy_pos[0]) ** 2 + (hydrogen_pos[1] - opposite_oxy_pos[1]) ** 2 +
                                (hydrogen_pos[2] - opposite_oxy_pos[2]) ** 2)

                            if hydrogen_distance < cutoff and has_covalent_bond is True:
                                failed = True
                                return True
    return False


def start_filtering():
    cutoff = 4
    while cutoff <= 5:
        failed_positive, failed_negative = 0,0
        with open("ho_bond_50.csv", newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter= ',', quotechar='|')
            row_count = 0
            for row in spamreader:
                if row_count == 0:
                    row_count+=1
                    continue
                if has_hydrogen_bond(row[1], cutoff):
                    failed_positive+=1
                if has_hydrogen_bond(row[3], cutoff):
                    failed_negative+=1
                row_count+=1
            print("Cutoff:",cutoff, " Bonded Positive:", failed_positive, " Bonded Negative:", failed_negative)
            cutoff+=0.02



# Args parser, run 1 smile string at a time
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles_string",
                        help="The smile string you would like to use",
                        type=str,
                        default=""
                        )
    return parser.parse_args()

def main():
    args = parse_args()
    #if args[0] == "":
    #    raise ValueError("First arg must be a single smile string")
    start_filtering()


if __name__ == "__main__":
    main()



