#!/usr/bin/env python3
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
from Bio.PDB import *
import hdbscan
from argparse import ArgumentParser

# add arguments
def parse_args():
    p = ArgumentParser(description=__doc__)
   # p.add_argument("--coord", help="retrieve water coordinates from pdb file made from visualization.py")
   # p.add_argument("--kde", help="retrieve crude KDE values from npy file made from KDE_full_protein.py")
   # p.add_argument("-d", help="directory to write out the pdb of the center of water molecules") #d1 = '/wynton/home/rotation/lisaz/fraser_water/water_placement/test_residue/PDB_files_AA_HS'
   # p.add_argument("-d2", help="directory with pdb file of all water molecules") #d2 = '/wynton/home/rotation/lisaz/fraser_water/water_placement/output_new2/density_pdb'
   # p.add_argument("-d3", help="directory with npy file of KDE values") #d3 = '/wynton/home/rotation/lisaz/fraser_water/water_placement/output_new2'
   # p.add_argument("--norm", help="normalization value")
    p.add_argument("--pdb", help="name of the pdb file of the water molecule center")
    p.add_argument("--cut", help="kde cutoff to include certain percentage of the original data set")
    args = p.parse_args()
    return args


def find_kde_cutoff(values, cut):
    '''a function that returns a threshold for KDE such that the trimmed data set includes 15% of the original data set'''
    hist, bins = np.histogram(np.exp(values))
    cutoff = []
    for i in bins:
        if ((np.exp(values) > i).sum() / sum(hist)) < float(cut):
            cutoff.append(i)
    return cutoff[0]


def make_trim_df(density_all_pdb, kde, cutoff):
    '''a function that first makes the full dataframe, then trims it based on kde cutoff'''
    parser=PDBParser()
    io = PDBIO()
    os.chdir("/wynton/home/rotation/lisaz/fraser_water/water_placement/output_new_rotamer/density_pdb")
    struc = parser.get_structure('name', density_all_pdb)
    model = struc[0]
    coor_list = []
    for chain in model.get_list():
        for residue in chain.get_list():
            wat = residue["O"]
            coor_list.append(wat.get_coord())
    coor = np.array(coor_list)
    all_coor_kde = pd.DataFrame({'x': coor[:,0], 'y': coor[:,1], 'z': coor[:,2], 'KDE': np.exp(kde)})
    trim_coor_kde = all_coor_kde[all_coor_kde['KDE'] > cutoff]
    return trim_coor_kde


def HDBSCAN(trim_coor_kde):
    ''' perform density clustering on trimmed data frame'''
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
        gen_min_span_tree=False, leaf_size=40, 
        metric='euclidean', min_cluster_size=5, min_samples=None, p=None)
    clusterer.fit(trim_coor_kde)
    labels = clusterer.labels_
    # construct final big df
    labeled_df = pd.DataFrame({'x': trim_coor_kde['x'], 'y': trim_coor_kde['y'], 'z': trim_coor_kde['z'], 'KDE': trim_coor_kde['KDE'], 'label': labels})
    return labels, labeled_df


def find_cluster_center(labeled_df, labels):
    HOH = []
    for i in np.arange(max(labels) + 1):
        cluster = labeled_df[labeled_df['label'] == i].to_numpy()
        center_ind = np.argmax(cluster[:, 3])
        HOH_coord = (cluster[center_ind,0], cluster[center_ind,1], cluster[center_ind,2])
        HOH.append(HOH_coord)
    return HOH


def build_center_PDB(hoh_coor, file_name): #hoh_coor is a list of tuples of coordinates of water molecules
    listy_wat = {}
    count=1
    for i, (xi, yi, zi) in enumerate(hoh_coor):
        listy_wat[i] = ['HETATM', str(i+1), 'O', 'HOH', str(i+1), str(format(xi, '.3f')), str(format(yi, '.3f')), str(format(zi, '.3f')), '1.00', '1.00', 'O']
    os.chdir('/wynton/home/rotation/lisaz/fraser_water/water_placement/output_new_rotamer/water_cluster_center')
    file = open(f'water_cluster_center_{file_name}_.pdb', "w")
    for row in list(listy_wat.values()):
        file.write("{: >1} {: >4} {: >2} {: >5} {: >5} {: >11} {: >7} {: >7} {: >5} {: >5} {: >11}\n".format(*row))
    file.close()
    


def main():
    args = parse_args()
    #os.chdir("/wynton/home/rotation/lisaz/fraser_water/water_placement/output_new2/density_pdb")
    print( f'{args.pdb}_1.0_norm_prot_density_all.pdb')
    density_all_pdb = f'{args.pdb}_1.0_norm_prot_density_all.pdb'
    os.chdir("/wynton/home/rotation/lisaz/fraser_water/water_placement/output_new_rotamer")
    kde = np.load(f'{args.pdb}_1.0_norm_prot_density_all.npy')
    kde = kde[::(len(kde) // 10000 + 1)]
    cutoff_kde = find_kde_cutoff(kde, args.cut)
    trim = make_trim_df(density_all_pdb, kde, cutoff_kde)
    labels, labeled_df = HDBSCAN(trim)
    HOH = find_cluster_center(labeled_df, labels)
    build_center_PDB(HOH, args.pdb)   

if __name__ == '__main__':
    main()
