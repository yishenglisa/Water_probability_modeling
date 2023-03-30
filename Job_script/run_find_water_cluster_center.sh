#!/bin/bash
#$ -l h_vmem=10G
#$ -l mem_free=20G
#$ -t 1-110 
#$ -l h_rt=20:00:00
#$ -pe smp 8
#$ -R yes
#$ -V

export PATH="/wynton/home/rotation/lisaz/.conda/envs:$PATH"
source activate qfit

PDB_file=/wynton/home/rotation/lisaz/fraser_water/water_placement/rotamers/rotamer_ids.txt
base_dir= '/wynton/home/rotation/lisaz/fraser_water/water_placement/rotamers/rotamer_pdb'
pdb=$(cat $PDB_file | awk '{ print $1 }' |head -n $SGE_TASK_ID | tail -n 1)
echo 'PDB:' ${pdb}
cd $base_dir

# run the script
/wynton/home/rotation/lisaz/.conda/envs/qfit/bin/python /wynton/home/rotation/lisaz/fraser_water/water_placement/analysis/HDBSCAN.py --pdb $pdb --cut 0.5 
