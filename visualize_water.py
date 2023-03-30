#!/usr/bin/env python3
import numpy as np 
import pandas as pd 
from analysis_functions import *         
from qfit.structure import Structure 
import glob 
import matplotlib.pyplot as plt 
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D 
from argparse import ArgumentParser 
from DICT4A_test2 import DICT4A

def parse_args():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--band", help="band value")    
    p.add_argument("--pdb", help="Name of PDB")
    p.add_argument('--norm', help="normalization value")    
    p.add_argument("-d", help="directory with all water coord npy")
    args = p.parse_args()
    return args

def main():
    args = parse_args() 
    os.chdir('/wynton/home/rotation/lisaz/fraser_water/water_placement/output_new_rotamer')
    out_coords_all_KDE = np.load(args.pdb + '_all_out_coord_test_50000_wat_new_analysis.npy',allow_pickle='TRUE')
    os.chdir('/wynton/home/rotation/lisaz/fraser_water/water_placement/output_new_rotamer')
    den_all = np.exp(np.load(f'{args.pdb}_{args.band}_{args.norm}_density_all.npy', allow_pickle='TRUE'))
    os.chdir('/wynton/home/rotation/lisaz/fraser_water/water_placement/rotamers/rotamer_pdb')
    s = Structure.fromfile(args.pdb + '.pdb').reorder()
    #output density.pdb
    os.chdir(args.d)
    build_density_pdb(out_coords_all_KDE.reshape(-1, 3), f'{args.pdb}_{args.band}_{args.norm}_density_all.pdb', den_all)
    

if __name__ == '__main__':
    main() 
