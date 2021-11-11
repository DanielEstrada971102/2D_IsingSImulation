#! /bin/bash

#PBS -N /scratch/gfif-user0/DanielE/data/timess
#PBS -o /scratch/gfif-user0/DanielE/data/timess.out
#PBS -e /scratch/gfif-user0/DanielE/data/timess.err
#PBS -l walltime=08:00:00
#PBS -M daniel.estrada1@udea.edu.co
#PBS -q long
#PBS -m e
#PBS -m a

time python3 /scratch/gfif-user0/DanielE/run.py -outfolder "/scratch/gfif-user0/DanielE/data"