import argparse
import os
import numpy as np
import pandas as pd

# Example input: python ds_creator.py 100 -eiso -riso -vert
# Parameters after molecule count specify which labels/outputs to use. Assumes 1 task regression model.

# currently named Azo_Data_1 on cluster
start_dir = "/cluster/research-groups/kowalczyk/stf_screen_cluster/Azo_Data_1"
#start_dir = "./Data"


# Driver method for multi-output regression dataset creation.
def start_ds_creation(args):
    # create initial np arrays
    id_arr = np.zeros(args.count, dtype=int)
    smiles_arr = []
    smiles_arr = ["hi" for i in range(args.count)]
    output_count = 0
    failed_arr = []
    if args.eisomerization:
        output_count += 1
    if args.reverse_isomerization:
        output_count += 1
    if args.vertical_excitation:
        output_count += 1
    if output_count == 0:
        raise ValueError('You must specify 1 output field. output_count=0')
    # weight_arr = np.ones([args.count, output_count], dtype=float)
    output_arr = np.zeros([output_count, args.count], dtype=float)
    mol_count = 0
    directory_list = os.listdir(start_dir)
    for batch in directory_list:
        batch_path = os.path.join(start_dir, batch)
        mol_list = os.listdir(batch_path)

        for molecule in mol_list:
            mol_path = os.path.join(batch_path, molecule)
            print(mol_path)
            if os.path.isdir(mol_path):
                # Pass smile string to featurizer, then add to input
                smiles_arr[mol_count] = get_smiles(mol_path, molecule)
                id_arr[mol_count] = molecule
                output_count = 0
                neb_path = get_neb_path(mol_path)
                if not os.path.isdir(neb_path):
                    failed_arr.append(molecule)
                    continue
                if args.reverse_isomerization:
                    output_arr[output_count, mol_count] = au_to_ev(get_barrier_height(neb_path))
                    output_count += 1
                if args.eisomerization:
                    output_arr[output_count, mol_count] = au_to_ev(get_meta_energy_dif(neb_path))
                    output_count += 1
                if args.vertical_excitation:
                    rel_gs_folder = get_gs_ex_path(mol_path, "gs")
                    rel_ex_folder = get_gs_ex_path(mol_path, "ex")
                    gs = get_total_electronic(rel_gs_folder)
                    ex = get_total_electronic(rel_ex_folder)
                    total_electronic_difference = ex - gs
                    output_arr[output_count, mol_count] = au_to_ev(total_electronic_difference)
                    output_count += 1

                mol_count += 1
                print("\n")
            else:
                failed_arr.append(molecule)
                print("Invalid molecule filepath " + mol_path + " (This may not be a folder)\n")
            if mol_count >= args.count:
                break
        if mol_count >= args.count:
            break

    create_save_dataset(id_arr, smiles_arr, output_arr, output_count, failed_arr)



# Create and save the dataset. Weight vector to be added here
# Featurizing should be done in ML model, deepchem CSVLoader class
def create_save_dataset(id, smiles_arr, output_arr, output_count, failed_arr):
    if output_count == 1:
        df = pd.DataFrame(list(zip(id, smiles_arr, output_arr[0])), columns=["ids", "smiles", "task1"])
    elif output_count == 2:
        df = pd.DataFrame(list(zip(id, smiles_arr, output_arr[0], output_arr[1])), columns=["ids", "smiles", "task1", "task2"])
    elif output_count == 3:
        df = pd.DataFrame(list(zip(id, smiles_arr, output_arr[0], output_arr[1], output_arr[2])), columns=["ids", "smiles", "task1", "task2", "task3"])
    df.to_csv('dataset_2000.csv')
    failed_df = pd.DataFrame(list(zip(failed_arr)), columns=["Failed Molecules"])
    failed_df.to_csv("failed_molecules")
    print("DONE!")


# Convert atomic units to electron volt
def au_to_ev(au):
    return 27.211324570273 * au


# Get the path to the stable excited and ground state dftb folders.
# gs_or_ex parameter is a string "gs" or "ex"
def get_gs_ex_path(molecule_path, gs_or_ex):
    meta = "Stable"
    dftb = "dftb"
    rel_meta = os.path.join(molecule_path, meta)
    rel_dftb = os.path.join(rel_meta, dftb)
    rel_folder = os.path.join(rel_dftb, gs_or_ex)
    return rel_folder


# Get electronic energy difference from detailed.out in gs or ex folder
# This is found in detailed.out (may need to come from neb folder instead)
def get_total_electronic(folder_path):
    filename = "detailed.out"
    rel_path = os.path.join(folder_path, filename)
    if os.path.isfile(rel_path):
        f = open(os.path.realpath(rel_path), "r")
        all_lines = f.readlines()
        line = all_lines[7]
        total_electronic = float(line[7:-34])
        #print(total_electronic)
        return total_electronic


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
        #print(energy)
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
        # print(f.readline())
        str_lines = f.readlines()
        one_line = str_lines[0]
        float_num = float(one_line)
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
        all_lines = f.readlines()
        smile = all_lines[0]
        print(smile)
        return smile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("count",
                        help="Specify how many molecules to use in dataset.",
                        type=int,
                        default=5
                        )
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
    parser.add_argument("-vert",
                        "--vertical_excitation",
                        help="Add vertical excitation as output task",
                        action="store_true"
                        )
    return parser.parse_args()


def main():
    args = parse_args()
    start_ds_creation(args)


if __name__ == "__main__":
    main()
