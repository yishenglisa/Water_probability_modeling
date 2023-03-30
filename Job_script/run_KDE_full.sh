#!/bin/bash
#$ -l h_vmem=10G
#$ -l mem_free=20G
#$ -t 1-155 
#$ -l h_rt=20:00:00
#$ -pe smp 8
#$ -R yes
#$ -V

export PATH="/wynton/home/rotation/lisaz/.conda/envs:$PATH"
source activate qfit


#cd /wynton/home/rotation/lisaz/fraser_water/water_placement
PDB_file=/wynton/home/rotation/lisaz/fraser_water/water_placement/rotamers/rotamer_ids.txt
base_dir= '/wynton/home/rotation/lisaz/fraser_water/water_placement/rotamers/rotamer_pdb'
#bandwidth=/wynton/home/rotation/lisaz/fraser_water/water_placement/test_residue/bandwidths.txt

pdb=$(cat $PDB_file | awk '{ print $1 }' |head -n $SGE_TASK_ID | tail -n 1)
echo 'PDB:' ${pdb}
cd $base_dir
#cd $pdb

#run intermediate output
#/wynton/home/rotation/lisaz/.conda/envs/qfit/bin/python /wynton/home/rotation/lisaz/fraser_water/water_placement/analysis/place_all_full_protein.py --pdb $pdb --norm 'norm_prot'

#cd /wynton/rotation/lisaz/fraser_water/water_placement/all_protein
#for i in {1..15}; do
 #   band=$(cat $bandwidth | awk '{ print $1 }' |head -n $i | tail -n 1)
#/wynton/home/rotation/lisaz/.conda/envs/qfit/bin/python /wynton/home/rotation/lisaz/fraser_water/water_placement/analysis/KDE_full_protein.py --pdb $pdb --band 1.0 -d /wynton/home/rotation/lisaz/fraser_water/water_placement/output_new_rotamer --norm 'norm_prot'
/wynton/home/rotation/lisaz/.conda/envs/qfit/bin/python /wynton/home/rotation/lisaz/fraser_water/water_placement/analysis/visualize_water.py --pdb $pdb -d /wynton/home/rotation/lisaz/fraser_water/water_placement/output_new_rotamer/density_pdb --norm 'norm_prot' --band 1.0

