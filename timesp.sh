#! /bin/bash

"/scratch/gfif-user0/DanielE/data/"
#PBS -N /scratch/gfif-user0/DanielE/data/timesp
#PBS -o /scratch/gfif-user0/DanielE/data/timesp.out
#PBS -e /scratch/gfif-user0/DanielE/data/timesp.err
#PBS -l walltime=04:00:00
#PBS -M daniel.estrada1@udea.edu.co
#PBS -m e
#PBS -m a

time python3 /scratch/gfif-user0/DanielE/run_p.py -outfolder "/scratch/gfif-user0/DanielE/data/"