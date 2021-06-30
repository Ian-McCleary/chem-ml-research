import argparse
import os

start_dir = "./Data/"

for root, subdirectories, files in os.walk(start_dir):
    for subdirectory in subdirectories:
        directory_path = os.path.join(root,subdirectory)
        batch,folder_name = os.path.splitext(directory_path)
        print(batch)
        print(folder_name)
        for file in files:
            rel_path1 = os.path.relpath(os.path.join(root, subdirectory))
            rel_path2 = os.path.join(rel_path1,file)
            print(rel_path2)
            name, extension = os.path.splitext(rel_path2)
            if "smiles" in extension:
                #print("found!")
                #f = open(os.path.realpath(rel_path2), "r")
                #print(f.readline())
