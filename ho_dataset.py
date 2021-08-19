import numpy as np
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
import math
from rdkit import RDLogger
import pandas as pd
import csv
import os

from ho_test import has_covalent_hydrogen_bond
from ho_test import find_half


def potential_hydrogen_bonding(smile):
    m = Chem.MolFromSmiles(smile)
    m = Chem.AddHs(m)
    status = AllChem.EmbedMolecule(m)
    conformer = m.GetConformer()

    atom_list = m.GetAtoms()
    bond_list = m.GetBonds()
    for i in range(len(atom_list)):
        if atom_list[i].GetSymbol() == "O":
            o1_tracking = []
            o_half1 = find_half(atom_list, bond_list, i, o1_tracking)
            if has_covalent_hydrogen_bond(i, atom_list, bond_list):
                for j in range(len(atom_list)):
                    if atom_list[j].GetSymbol() == "O":
                        o2_tracking = []
                        o_half2 = find_half(atom_list, bond_list, j, o2_tracking)
                        if (o_half1 is True and o_half2 is False) or (o_half1 is False and o_half2 is True):
                            return True
    return False


def get_mol_from_xyz(mol_id):
    start_dir = "/cluster/research-groups/kowalczyk/stf_screen_cluster/Azo_Data_1"
    reversed_line_arr = []
    directory_list = os.listdir(start_dir)
    for batch in directory_list:
        batch_path = os.path.join(start_dir, batch)
        potential_mol_path = os.path.join(batch_path,mol_id)
        if os.path.isdir(potential_mol_path):
            xyz_path_str = "Meta/dftb/gs/opt.xyz"
            xyz_path = os.path.join(potential_mol_path, xyz_path_str)
            if os.path.isfile(xyz_path):
                stop = False
                for line in reversed(open(xyz_path).readlines()):
                    split_line = line.split("\n")
                    reversed_line_arr.append(split_line[0])
                    if stop is False:
                        if "Geometry" in line:
                            stop = True
                    else:
                        break
    for i in range(len(reversed_line_arr)):
        print(reversed_line_arr[len(reversed_line_arr)-i-1])
    #print(reversed_line_arr)




def start_creation():
    neg_list,pos_list,neg_iso,pos_iso = [],[],[],[]
    
    # Negative Isomerization Energy
    with open("failed_molecules.csv", newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter= ',', quotechar='|')
        mol_count = 0
        for row in spamreader:
            if mol_count > 10:
                break
            elif mol_count == 0:
                mol_count+=1
                continue
            get_mol_from_xyz(row[1])
            if potential_hydrogen_bonding(row[3]):

                neg_list.append(row[3])
                neg_iso.append(row[4])
                mol_count +=1
    
    # Positive isomerization energy
    with open("Datasets/dataset_3task_50k_filtered.csv", newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter= ',', quotechar='|')
        mol_count1 = 0
        for row in spamreader:
            #print(row[2])
            if mol_count1 > 10:
                break
            elif mol_count1 == 0:
                mol_count1+=1
                continue
            if potential_hydrogen_bonding(row[2]):
                pos_list.append(row[2])
                pos_iso.append(row[4])
                mol_count1 +=1
    
    df = pd.DataFrame(list(zip(pos_list, pos_iso, neg_list, neg_iso)),columns=["Positive_Smile", "Positive_Iso","Negative_Smile", "Negative_Iso"])

    df.to_csv("ho_bond_50.csv")


start_creation()