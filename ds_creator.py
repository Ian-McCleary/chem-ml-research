import argparse
import os

start_dir = "./Data/"

directory_list = os.listdir(start_dir)
for batch in directory_list:
    batch_path = os.path.join(start_dir, batch)
    mol_list = os.listdir(batch_path)
    for mol in mol_list:
        mol_path = os.path.join(batch_path, mol)
        if os.path.isdir(mol_path):
            smile_append = "/Meta/" + mol + "_Meta.smiles"
            smile_rel_path = os.path.join(mol_path, smile_append)
            if os.path.isfile(smile_rel_path):
                print("found!")
                f = open(os.path.realpath(smile_rel_path), "r")
                print(f.readline())




            #rel_path1 = os.path.relpath(os.path.join(root, subdirectory))
            #rel_path2 = os.path.join(rel_path1,file)
            #name, extension = os.path.splitext(rel_path2)
            #if "smiles" in extension:
             #   print("found!")
                #f = open(os.path.realpath(rel_path2), "r")
                #print(f.readline())
