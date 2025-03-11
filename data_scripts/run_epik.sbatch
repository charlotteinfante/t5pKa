#!/bin/bash
#SBATCH --job-name=epik 
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1 
#SBATCH --time=0:30:00
#SBATCH --mem=20GB 
#SBATCH --gres=gpu:1

cd /scratch/cii2002/

module purge
module load schrodinger/2021-1
ligprep -s 1 -t 1 -i 0 -ismi /scratch/cii2002/t5chem_new/t5chem_prop/pka/data/ -omae mae_file_name -WAIT

epik -scan -imae /scratch/cii2002/testing_epik.mae -omae /scratch/cii2002/predict_epik_testing.mae -ph 7.4 -p 0.1 -highest_pka 14.0 -lowest_pka 0.0 -WAIT

sdconvert -imae /scratch/cii2002/predict_epik_testing.mae -osdf predict_epik_testing.sdf -annstereo -pKa
