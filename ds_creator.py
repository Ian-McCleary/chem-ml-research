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
            #print(mol_path)
            smile_append_str = "/Meta/" + mol + "_Meta.smiles"
            smile_append_path = os.path.normpath(smile_append_str)
            smile_rel_path = os.path.join(mol_path, smile_append_path)
            smile_absolute = os.path.realpath(smile_rel_path)
            print(smile_rel_path)
            if os.path.isfile(smile_absolute):
                print("found!")
                f = open(os.path.realpath(smile_rel_path), "r")
                print(f.readline())
            else:
                print("not a smile path")
        else:
            print("not a mol_path")


            #rel_path1 = os.path.relpath(os.path.join(root, subdirectory))
            #rel_path2 = os.path.join(rel_path1,file)
            #name, extension = os.path.splitext(rel_path2)
            #if "smiles" in extension:
             #   print("found!")
                #f = open(os.path.realpath(rel_path2), "r")
                #print(f.readline())
