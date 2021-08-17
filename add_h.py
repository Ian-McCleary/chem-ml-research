import numpy as np
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
import math
from rdkit import RDLogger
import pandas as pd
import csv

for x in range(3):
    print("Batch:",x,"\n")
    with open("failed_molecules.csv", newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter= ',', quotechar='|')
            mol_count = 0
            for row in spamreader:
                if mol_count == 0:
                    mol_count+=1
                    continue
                print("Molecule:",mol_count)
                smile = row[3]
                m = Chem.MolFromSmiles(smile)
                m = Chem.AddHs(m)
                status = AllChem.EmbedMolecule(m)
                conformer = m.GetConformer()
                pos = conformer.GetPositions()

                atom_list = m.GetAtoms()
                for i in range(len(atom_list)):
                    if atom_list[i].GetSymbol() == "H":
                        h_pos = pos[i]
                        print(i, "x:",h_pos[0],"y:",h_pos[1],"z:",h_pos[2])
                mol_count+=1
                if mol_count >4:
                    break
        
