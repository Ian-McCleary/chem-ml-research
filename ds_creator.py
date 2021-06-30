import argparse
import os

start_dir = "./Data/"

for root, subdirectories, files in os.walk(start_dir):
    for subdirectory in subdirectories:
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, subdirectory))
            print(rel_path)
            name, extension = os.path.splitext(rel_path)
            if "smiles" in extension:
                print("found!")
