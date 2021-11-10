#! /bin/bash
#PBS -N program
#PBS -o program.out
#PBS -e program.err
#PBS -l walltime=02:00:00
#PBS -M daniel.estrada1@udea.edu.co
#PBS -m e
#PBS -m a

time python3 /scratch/gfif-user0/DanielE/run_transition.py