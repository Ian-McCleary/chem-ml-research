import argparse
import os
import numpy as np

# currently named Azo_Data_1 on cluster
start_dir = "./Data/"


# TODO Add parameters which allow specific data collection fields. Remove print statements. Add variables to dataset object type & write to file
# Driver method for dataset creation.
def start_ds_creation(args):
    #create initial np arrays
    id_arr = np.zeros([args.count])

    # For coulomb matrix, input_arr must be [args.count, max_atoms, max_atoms]
    #input_arr = np.zeros([args.count])
    input_arr = np.zeros[args.count]

    if args.eisomerization:
        eiso_arr = np.zeros([args.count])
    if args.reverse_isomerization:
        riso_arr = np.zeros([args.count])
    if args.vertical_excitation:
        vexci_arr = np.zeros([args.count])
    if args.internal_conversion:
        inc_arr = np.zeros([args.count])

    mol_count = 0
    directory_list = os.listdir(start_dir)
    for batch in directory_list:
        batch_path = os.path.join(start_dir, batch)
        mol_list = os.listdir(batch_path)

        for molecule in mol_list:
            mol_path = os.path.join(batch_path, molecule)
            if os.path.isdir(mol_path):
                print(args.count)
                print(len(input_arr))
                print(mol_count)
                #Pass smile string to featurizer, then add to input
                input_arr[mol_count] = get_smiles(mol_path, molecule)
                id_arr[mol_count] = molecule

                neb_path = get_neb_path(mol_path)
                if args.reverse_isomerization:
                    riso_arr[mol_count] = get_barrier_height(neb_path)
                if args.eisomerization:
                    eiso_arr[mol_count] = get_meta_energy_dif(neb_path)
                if args.vertical_excitation:
                    rel_gs_folder = get_gs_ex_path(mol_path, "gs")
                    get_electronic_dif(rel_gs_folder)

                mol_count += 1
                print("\n")
            else:
                print("Invalid molecule filepath " + mol_path + " (This may not be a folder)")
            if mol_count >= args.count:
                break
        if mol_count >= args.count:
            break

    print(id_arr)
    print(eiso_arr)
    print(riso_arr)


# Get the path to the excited and ground state dftb folders.
# gs_or_ex parameter is a string "gs" or "ex"
def get_gs_ex_path(molecule_path, gs_or_ex):
    meta = "Meta"
    dftb = "dftb"
    rel_meta = os.path.join(molecule_path, meta)
    rel_dftb = os.path.join(rel_meta, dftb)
    rel_folder = os.path.join(rel_dftb, gs_or_ex)
    return rel_folder


# Get electronic energy difference from detailed.out in gs or ex folder
# This is found in detailed.out (may need to come from neb folder instead)
def get_electronic_dif(folder_path):
    filename = "detailed.out"
    rel_path = os.path.join(folder_path,filename)
    if os.path.isfile(rel_path):
        f = open(os.path.realpath(rel_path), "r")
        all_lines = f.readlines()
        line = all_lines[7]
        diff_electronic = float(line[25:-17])
        print(diff_electronic)
        return diff_electronic

# Energy difference between stable and metastable found in neb.out
def get_meta_energy_dif(neb_path):
    out = "neb.out"
    rel_neb_out = os.path.join(neb_path, out)
    if os.path.isfile(rel_neb_out):
        f = open(os.path.realpath(rel_neb_out), "r")
        all_lines = f.readlines()
        stable_line = all_lines[3]
        meta_line = all_lines[5]
        stable_float = float(stable_line[23:-7])
        meta_float = float(meta_line[23:-7])
        energy = meta_float - stable_float
        print(energy)
        print(stable_float)
        return energy

# Create the path for neb_out string. Must use os.path.join() and cannot use string + operator
def get_neb_path(rel_molecule_path):
    neb1 = "neb"
    neb2 = "neb_out"
    rel_neb = os.path.join(rel_molecule_path, neb1)
    rel_neb_out = os.path.join(rel_neb, neb2)
    return rel_neb_out

# Get barrier height found in neb folder
def get_barrier_height(neb_path):
    b_file = "barrier.height"
    barrier_path = os.path.join(neb_path, b_file)
    if os.path.isfile(barrier_path):
        f = open(os.path.realpath(barrier_path), "r")
        #print(f.readline())
        str_lines = f.readlines()
        one_line = str_lines[0]
        str_num = float(one_line[:-1])
        if one_line.isnumeric():
            print("true")
        else:
            print("false")
        float_num = float(str_num)
        print(float_num)
        return float_num


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
        smile = f.readline()
    else:
        print("Invalid smiles filepath")
    return smile

# Allows for custom featurizer code for specific models
def featurize_smiles(smile_string):
    # add deepchem featurizer code here such as Coulombmatrix ect..
    feature = smile_string
    return feature

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("count",
                        help="Specify how many molecules to use in dataset.",
                        type=int,
                        default=5)
    parser.add_argument("-eiso",
                        "--eisomerization",
                        help="Add delta E_iso as output task",
                        action="store_true"
                        )
    parser.add_argument("-riso",
                        "--reverse_isomerization",
                        help="Add reverse isomerization as output task",
                        action="store_true"
                        )
    parser.add_argument("-vexci",
                        "--vertical_excitation",
                        help="Add vertical excitation as output task",
                        action="store_true"
                        )
    parser.add_argument("-iconv",
                        "--internal_conversion",
                        help="Add internal conversion as output task",
                        action="store_true"
                        )
    return parser.parse_args()


def main():
    args = parse_args()
    start_ds_creation(args)


if __name__ == "__main__":
    main()
