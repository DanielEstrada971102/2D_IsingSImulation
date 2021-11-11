#! /bin/bash

#PBS -N /scratch/gfif-user0/DanielE/data/transition
#PBS -o /scratch/gfif-user0/DanielE/data/transition.out
#PBS -e /scratch/gfif-user0/DanielE/data/transition.err
#PBS -l walltime=04:00:00
#PBS -M daniel.estrada1@udea.edu.co
#PBS -m e
#PBS -m a

time python3 /scratch/gfif-user0/DanielE/run_transition_p.py -outfolder "/scratch/gfif-user0/DanielE/data/" -L 256