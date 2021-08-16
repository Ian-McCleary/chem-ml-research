import numpy as np
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
import math
from rdkit import RDLogger
import pandas as pd
import csv


def potential_hydrogen_bonding(smile):
    m = Chem.MolFromSmiles(smile)
    m = Chem.AddHs(m)
    status = AllChem.EmbedMolecule(m)
    conformer = m.GetConformer()

    atom_list = m.GetAtoms()
    bond_list = m.GetBonds()
    for i in range(len(atom_list)):
        if atom_list[i].GetSymbol() == "O":
            # Check if oxygen has covalent hydrogen bond
            for j in range(len(bond_list)):
                try:
                    connecting_atom = Chem.rdchem.Bond.GetOtherAtomIdx(bond_list[j], i)
                except (RuntimeError):
                    continue
                if atom_list[connecting_atom].GetSymbol() == "H":
                    return True
    return False


def start_creation():
    neg_list,pos_list,neg_iso,pos_iso = [],[],[],[]
    # Negative Isomerization Energy
    with open("failed_molecules.csv", newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter= ',', quotechar='|')
        mol_count = 0
        for row in spamreader:
            if mol_count > 50:
                break
            elif mol_count == 0:
                mol_count+=1
                continue
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
            if mol_count1 > 50:
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