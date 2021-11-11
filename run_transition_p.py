""" ****************************************************************************
* @file run.py                                                                 *
* @author Daniel Estrada (daniel.estrada1@udea.edu.co)                         *
* @brief Runs the 2D ising  model simulation for several lattice size values,  *
*       generates the files with the data for compute magnetization and the    *
*       save the computing time expended                                       *
* @version 0.2                                                                 *
* @date 2021-11-8                                                              *                                     
*                                                                              *
* @copyright Copyright (c) 2021                                                *
********************************************************************************
"""
from multiprocessing import Pool, cpu_count
from ising_classes import * # importing the classes for the simulation
from numpy import arange


def run_simulation(b, L, N, init_cold = True, vb=True):
    """
    This function generates N updates of a LxL system for extract the values 
    of magnetization in every step for a temperature "b" and 
    saves the data in a file. 

    @Parameters
    -----------
    ->  L : int
        ... length of the particles lattice
    ->  N : int
        ... sample size to compute the monte carlo estimator
    ->  m : int
        ... number of Metropolis-hasting algotihm repetitions
    ->  init_cold : bool
        ... if true, the sistem will start in a cold state (every spins up (1))
    ->  vb : bool
        ... verbose argument
    ->  b : float 
        ... beta parameter for the simulation
    @Retuns
    -------
    -
    """

    if vb : print("runing for b = %.3f"%(b))

    M = []
    system = IsingLattice(L, b, init_cold) # the particles sysyem is initialized
        
    # initial magnetization and energy values are written
    M.append(system.M)

    for n in range(1, N):         
        # a new spins configuration is genereted with Metropolis algorithm          
        system.updateLattice(L*L)            
        # new magnetization and energy values are written
        M.append(system.M)

    M_mc = sum(M[20:])/len(M[20:]) 

    return (b, M_mc)

def main(L, N, outfolder, cold=True, verbose=True):   
    """
    run the simulation for several beta values for a size L.
    
    @Parameters
    -----------
    ->  L : float 
        ... Lattice Size
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

    betas = arange(0.01, 1, 0.01)

    filename = outfolder + "transitionp.txt"
    
    if verbose: print("runing in several betas in parallel...")

    with open(filename, 'w') as file: # file to save the simulations times
        file.write("b \t M\n")
        
        pool = Pool(cpu_count())
        
        results = [pool.apply_async(run_simulation, args=[b, L, N, cold, verbose]) for b in betas]

        for r in results:
            b , M = r.get()
            file.write("%.3f \t %.3f\n"%(b, M))

if __name__ == "__main__":
    #---------------------------- Args configurations ------------------------------      
    parser = argparse.ArgumentParser(description="This program runs the 2D ising\
    model simulation for differents lattice size, it generates the files with the \
    data for compute magnetization")

    parser.add_argument('-L', type=int, metavar='', default=10,
                    help='Lattice size')
    parser.add_argument('-N', type=int, metavar='', default=1000, 
                    help='number of updates of the lattice')
    parser.add_argument('-outfolder', type=str, metavar='', default="data/", 
                    help='path to save the output file')                
    parser.add_argument('-cold', type=int, metavar='', default=1, choices=[0,1],
                    help='if 1, init the lattice in cold state')
    parser.add_argument('-nv', type=int, metavar='', default=0, choices=[0,1],
                    help='verbose disable')

    args = parser.parse_args()
    # main execution...........
    main(args.L, args.N, args.outfolder, args.cold, not args.nv)