import argparse
import os

#currently named Azo_Data_1 on cluster
start_dir = "./Data/"

# TODO Add parameters which allow specific data collection fields and create customizable featurizer function
def start_ds_creation():
    directory_list = os.listdir(start_dir)
    for batch in directory_list:
        batch_path = os.path.join(start_dir, batch)
        mol_list = os.listdir(batch_path)
        for mol in mol_list:
            mol_path = os.path.join(batch_path, mol)
            if os.path.isdir(mol_path):
                get_smiles(mol_path, mol)
            else:
                print("Invalid molecule filepath" + mol_path)



def get_smiles(rel_molecule_path, molecule_name):
    # Create the path for smile string. Must use os.path.join() and cannot use string + operator
    # Using Meta smile string
    sstr1 = "Meta"
    s_append1 = os.path.join(rel_molecule_path, sstr1)
    s_filename = molecule_name + "_Meta.smiles"
    # smile_append_path = os.path.normpath(smile_append_str)
    smile_rel_path = os.path.join(s_append1, s_filename)
    smile_absolute = os.path.realpath(smile_rel_path)
    print(smile_rel_path)
    if os.path.isfile(smile_rel_path):
        f = open(os.path.realpath(smile_rel_path), "r")
        print(f.readline())
    else:
        print("Invalid smiles filepath")


def main():
    start_ds_creation()

if __name__ == "__main__":
    main()

