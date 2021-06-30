import argparse
import os

start_dir = "./Data/"

for root, subdirectories, files in os.walk(start_dir):
    for subdirectory in subdirectories:
        for file in files:
            rel_path1 = os.path.relpath(os.path.join(root, subdirectory))
            rel_path2 = os.path.join(rel_path1,file)
            print(rel_path2)
            name, extension = os.path.splitext(rel_path2)
            if "smiles" in extension:
                print("found!")
                f = open(file, "r")
                print(f.readline())
