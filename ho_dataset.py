import numpy as np
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem
import math
from rdkit import RDLogger
import deepchem as dc
import numpy as np
import pandas as pd
import tensorflow as tf
import csv

with open("failed_molecules.csv", newline='') as csvfile:
    spamreade = csv.reader(csvfile, delimiter= ' ', quotechar='|')
    for row in spamreader:
        print(row[3])

