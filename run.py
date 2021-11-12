""" ****************************************************************************
* @file run.py                                                                 *
* @author Daniel Estrada (daniel.estrada1@udea.edu.co)                         *
* @brief Runs the 2D ising  model simulation for several lattice size values,  *
*       generates the file of the computing time expended                      *
* @version 0.2                                                                 *
* @date 2021-11-8                                                              *                                     
*                                                                              *
* @copyright Copyright (c) 2021                                                *
********************************************************************************
"""
from time import perf_counter
from tqdm import tqdm
from ising_classes import * # importing the classes for the simulation


def main(b, N, cold=True, verbose=True):   
    """
    the simulation time expended  for each L is computed and saved in a file
    
    @Parameters
    -----------
    ->  b : float 
        ... beta parameter for the simulation
    ->  N : int
        ... sample size to compute the monte carlo estimator
    ->  cold : bool
        ... if true, the sistem will start in a cold state (every spins up (1))
    ->  verbose : bool
        ... verbose argument
    @Retuns
    -------
    -
    """
    
    sizes = [15, 30, 60, 120, 240, 480]
    
    filename = "tiempos_sec.txt"
    
    if verbose: print("runing in serial for b = %.3f ..."%b)

    with open(filename, 'w') as file: # file to save the simulations times
        file.write("L \t time(s)\n")
        
        for l in sizes:
            t_star = perf_counter() # start time
    
            if verbose : print("runing L = %d"%(l))
            
            system = IsingLattice(l, b, cold) # the particles sysyem is initialized
            for n in tqdm(range(1, N), disable = not verbose):         
                # a new spins configuration is genereted with Metropolis algorithm          
                system.updateLattice(l*l)

            t_end = perf_counter() # finish time
            
            if verbose : print(" time spended : ", t_end - t_star, " s")
            file.write("%d \t %.5f\n"%(l, t_end - t_star))


if __name__ == "__main__":
    #---------------------------- Args configurations ------------------------------      
    parser = argparse.ArgumentParser(description="This program runs the 2D ising\
    model simulation for differents lattice size and saves the execution times in\
    a file generates the files with the ")

    parser.add_argument('-b', type=float, metavar='', default=0.3,
                    help='temperature parameter')
    parser.add_argument('-N', type=int, metavar='', default=1000, 
                    help='number of updates of the lattice')  
    parser.add_argument('-cold', type=int, metavar='', default=1, choices=[0,1],
                    help='if 1, init the lattice in cold state')
    parser.add_argument('-nv', type=int, metavar='', default=0, choices=[0,1],
                    help='verbose disable')

    args = parser.parse_args()
    # main execution...........
    main(args.b, args.N, args.cold, not args.nv)