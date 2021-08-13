import numpy as np
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
import math
from rdkit import RDLogger
smiles = ["COc1cccc(\\N=N/c2c(C)ccc(C#N)c2C)c1C(=O)O", "COc1cccc(C(=O)O)c1\\N=N/c1ccc(C(=O)O)c(C(=O)O)c1C", "Cc1ccc(C(=O)O)c(\\N=N/c2cccc(F)c2C(=O)O)c1F",
"COc1c(C)cccc1\\N=N/c1c(-c2ccccc2)ccc(F)c1C#N", "COc1cccc(\\N=N/c2cc(OC)c(C(=O)O)c(-c3ccccc3)c2)c1OC", "COc1cccc(\\N=N/c2cc(C#N)cc(OC)c2C(=O)O)c1C(=O)O"]

passed_smiles = ["COc1cc(\\N=N/c2c(C)ccc(-c3ccccc3)c2C#N)ccc1C(=O)O", "Cc1ccc(\\N=N/c2c(F)cc(C(=O)O)cc2C(=O)O)c(C(=O)O)c1", "COc1cc(C#N)ccc1\\N=N/c1cc(C)c(C)cc1-c1ccccc1",
                 "N#Cc1ccccc1\\N=N/c1cc(C(=O)O)ccc1-c1ccccc1", "O=C(O)c1cc(F)c(\\N=N/c2cccc(F)c2)c(C(=O)O)c1", "COc1ccc(OC)c(\\N=N/c2c(F)ccc(-c3ccccc3)c2C#N)c1"]


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


#Test all possible bond paths to the N=N bond.
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


def start_filtering():
    lg = RDLogger.logger()

    lg.setLevel(RDLogger.CRITICAL)
    # smiles = ["COc1cccc(\\N=N/c2ccc(-c3ccccc3)c(C)c2OC)c1C(=O)O"]
    for smile in passed_smiles:
        print(smile)
        m = Chem.MolFromSmiles(smile)
        m = Chem.AddHs(m)
        status = AllChem.EmbedMolecule(m)
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
                bonded_h_val = 0
                for j in range(len(atom_list)):
                    b_1 = atom_list[j]
                    if b_1.GetSymbol() == "H":

                        # recursively check the side of each hydrogen atom
                        track_list = []
                        near_o_c = find_nearest_oxygen_or_carbon(atom_list, bond_list, j, track_list)
                        track_list = []
                        h_half = find_half(atom_list, bond_list, near_o_c, track_list)
                        # print("answer: ", answer)
                        # print(o_half, h_half)
                        if (not o_half is True and h_half is True) or (not o_half is False and h_half is False):
                            failed = False
                            oxy_pos = pos[i]
                            hydro_pos = pos[j]
                            hydrogen_distance = math.sqrt(
                                (oxy_pos[0] - hydro_pos[0]) ** 2 + (oxy_pos[1] - hydro_pos[1]) ** 2 +
                                (oxy_pos[2] - hydro_pos[2]) ** 2)
                            # print(distance)
                            if hydrogen_distance < 4.2 and has_covalent_bond is True:
                                failed = True
                                # nearby_o = nearest_oxygen_distance(atom_list, pos, oxy_pos)
                                # if nearby_o < hydrogen_distance and nearby_o < 2.3:
                                #    failed = False
                                # else:
                                #    failed = True
                            if failed == True:
                                print("Failed: ", oxy_count, "  ", hydrogen_distance)
                                break


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



