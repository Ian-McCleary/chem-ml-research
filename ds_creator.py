import argparse
import os

start_dir = "./Data/"

directory_list = os.listdir(start_dir)
print(directory_list)


            #rel_path1 = os.path.relpath(os.path.join(root, subdirectory))
            #rel_path2 = os.path.join(rel_path1,file)
            #name, extension = os.path.splitext(rel_path2)
            #if "smiles" in extension:
             #   print("found!")
                #f = open(os.path.realpath(rel_path2), "r")
                #print(f.readline())
