import argparse
import os

# currently named Azo_Data_1 on cluster
start_dir = "./Data/"


# TODO Add parameters which allow specific data collection fields. Remove print statements. Add variables to dataset object type
def start_ds_creation():
    directory_list = os.listdir(start_dir)
    for batch in directory_list:
        batch_path = os.path.join(start_dir, batch)
        mol_list = os.listdir(batch_path)
        for molecule in mol_list:
            mol_path = os.path.join(batch_path, molecule)
            if os.path.isdir(mol_path):
                get_smiles(mol_path, molecule)
                neb_path = get_neb_path(mol_path)
                get_barrier_height(neb_path)
                get_meta_energy_dif(neb_path)
                print("\n")
            else:
                print("Invalid molecule filepath" + mol_path + " This may not be a folder.")


def get_meta_energy_dif(neb_path):
    out = "neb.out"
    rel_neb_out = os.path.join(neb_path, out)
    if os.path.isfile(rel_neb_out):
        f = open(os.path.realpath(rel_neb_out), "r")
        all_lines = f.readlines()
        stable_line = all_lines[2]
        meta_line = all_lines[4]
        #stable_float = float(stable_line[23:-7])
        #meta_float = float(meta_line[23:-7])
        #energy = meta_float - stable_float
        print(stable_line[23:-7])
        #return energy

# Create the path for neb_out string. Must use os.path.join() and cannot use string + operator
def get_neb_path(rel_molecule_path):
    neb1 = "neb"
    neb2 = "neb_out"
    rel_neb = os.path.join(rel_molecule_path, neb1)
    rel_neb_out = os.path.join(rel_neb, neb2)
    return rel_neb_out


def get_barrier_height(neb_path):
    b_file = "barrier.height"
    barrier_path = os.path.join(neb_path, b_file)
    if os.path.isfile(barrier_path):
        f = open(os.path.realpath(barrier_path), "r")
        print(f.readline())
        return f.readline()


def get_smiles(rel_molecule_path, molecule_name):
    # Create the path for smile string. Must use os.path.join() and cannot use string + operator
    # Using Meta smile string
    sstr1 = "Meta"
    s_append1 = os.path.join(rel_molecule_path, sstr1)
    s_filename = molecule_name + "_Meta.smiles"
    smile_rel_path = os.path.join(s_append1, s_filename)
    smile_absolute = os.path.realpath(smile_rel_path)
    if os.path.isfile(smile_rel_path):
        f = open(os.path.realpath(smile_rel_path), "r")
        print(f.readline())
    else:
        print("Invalid smiles filepath")

# Allows for custom featurizer code for specific models
def featurize_smiles(smile_string):
    # add deepchem featurizer code here such as Coulombmatrix ect..
    feature = smile_string
    return feature


def main():
    start_ds_creation()


if __name__ == "__main__":
    main()
