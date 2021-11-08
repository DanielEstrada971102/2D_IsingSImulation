""" ****************************************************************************
* @file run.py                                                      *
* @author Daniel Estrada (daniel.estrada1@udea.edu.co)                         *
* @brief Runs the 2D ising  model simulation, generates the files with the     *
*        data for compute magnetization and the system evolution visualization *
* @version 0.1                                                                 *
* @date 2021-10-14                                                             *                                     
*                                                                              *
* @copyright Copyright (c) 2021                                                *
********************************************************************************
"""
from os import system as sys
from numpy import arange
from tqdm import tqdm
from ising_classes import * # importing the classes for the simulation
from multiprocessing import Process


def run_simulation(b, N, L, m, folder = ".", init_cold = True, vb=True):
    """
    This function generates N updates of a LxL system for extract the values 
    of magnetization in every step for differents temperatures "betas" and 
    saves the data in "folder". 

    @Parameters
    -----------
    ->  betas : float list or array/arange : numpy
        ... list with the temperature variation for the simulation
    ->  N : int
        ... sample size to compute the monte carlo estimator
    ->  L : int
        ... length of the particles lattice
    ->  m : int
        ... number of Metropolis-hasting algotihm repetitions
    ->  folder : str
        ... directory path to save the output files
    ->  init_cold : bool
        ... if true, the sistem will start in a cold state (every spins up (1))
    ->  vb : bool
        ... verbose argument
    @Retuns
    -------
    -
    """

    if vb : print("runing for L = %.d"%L)


    #path to save the spins configuration file and visualize the sytem evolution
    spinConfigFiles_path = folder + "/visual/b_%.3f"%b
    #path to save the magnetization and energy data
    dataFiles_path = folder + "/data/b%.3f.txt"%b

    sys("mkdir " + spinConfigFiles_path) #creating the folder

    with open(dataFiles_path, 'w') as f:
        
        f.write("M\tE\n") #file header 
        if vb: print("runing for beta = %.3f"%b)
        
        system = IsingLattice(L, b) # the particles sysyem is initialized
        
        # initial magnetization and energy values are written
        f.write("%.4f\t%.4f\n"%(system.M, system.E)) 
        # initial spins configuration is saved
        gen_visualFile(spinConfigFiles_path + "/step0.txt"%b, system)

        for n in tqdm(range(1, N), disable = not vb):         
            # a new spins configuration is genereted with Metropolis algorithm          
            system.updateLattice(m)            
            # new system spins configuration is saved
            gen_visualFile(spinConfigFiles_path + "/step%d.txt"%(n), system)
            # new magnetization and energy values are written
            f.write("%.4f\t%.4f\n"%(system.M, system.E))


def gen_visualFile(filename, system):
    """
    This function writes into a file the system spins configuration

    @Parameters
    -----------
    ->  filename : str
        ... file name to save the spins configuration
    ->  system : IsingLatice : ising_classes
        ... particles system 
    @Retuns
    -------
    -
    """
    with open(filename, 'w') as f:
        for row in system.get_spinMatrix():
            line = ["%d\t"%spin for spin in row] 
            f.write(''.join(line)+"\n")


def main(bmin, bmax, bstep, N, l, m, attempt='', cold=True, verbose=True, 
         parallel=False):   
    # the container folders are created
    folderName = "L" + attempt + "_%d"%l
    sys("mkdir " + folderName)
    sys("mkdir " + folderName+"/data " + folderName + "/visual")

    betas = arange(bmin, bmax, bstep)

    # the simulation is runed
    if parallel:
        processes = []
        for b in betas:
            processes.append(Process(target=run_simulation, 
                            args=[b,N, l, m, folderName, cold, verbose]))
            processes[-1].start()
        
        for process in processes:
            process.join()
    
    else:
        for b in betas:
            run_simulation(b, N, l, m, folderName, cold, verbose)


if __name__ == "__main__":
    #---------------------------- Args configurations ------------------------------      
    parser = argparse.ArgumentParser(description="This program runs the 2D ising\
    model simulation, generates the files with the data for compute magnetization\
    and the system evolution visualization. the main function executs the simulation\
    with the parameters passed.")
    parser.add_argument('-L', type=int, metavar='', default=10, 
                    help='lenght of the lattice')
    parser.add_argument('-bmin', type=float, metavar='', default=0,
                    help='min beta for the simulation')
    parser.add_argument('-bmax', type=float, metavar='', default=1, 
                    help='max beta for the simulation')
    parser.add_argument('-bstep', type=float, metavar='', default=.01, 
                    help='beta step for the simulation')
    parser.add_argument('-N', type=int, metavar='', default=1000, 
                    help='number of updates of the lattice')
    parser.add_argument('-m', type=int, metavar='', default=100, 
                    help='number of repetitions to one update of the lattice')
    parser.add_argument('-cold', type=int, metavar='', default=1, choices=[0,1],
                    help='if 1, init the lattice in cold state')
    parser.add_argument('-attmp', type=str, metavar='', default='', 
                    help='If you have run more than once for this L, it is \
                          necessary to specify the number of the attempt')
    parser.add_argument('-nv', type=int, metavar='', default=0, choices=[0,1],
                    help='verbose disable')
    parser.add_argument('-p', type=int, metavar='', default=0, choices=[0,1],
                    help='run in parallel')

    args = parser.parse_args()
    # main execution...........
    main(args.bmin, args.bmax, args.bstep, args.N, args.L, 
        args.m, args.attmp, args.cold, not args.nv, args.p)