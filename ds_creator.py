import argparse
import os

start_dir = "./Data/"

for root, subdirectories, files in os.walk(start_dir):
    for subdirectory in subdirectories:
        for file in files:
            name, extension = os.path.splitext(os.path.relpath())
            if "smiles" in extension:
                print("found!")