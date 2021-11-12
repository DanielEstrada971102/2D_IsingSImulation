""" ****************************************************************************
* @file   run_transition_p.py                                                  *
* @author Daniel Estrada (daniel.estrada1@udea.edu.co)                         *
* @brief Runs the 2D ising  model simulation for several beta values in        *
*        parallel, generates the files with the data for compute phase         *
*        transition                                                            *
* @version 0.1                                                                 *
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
    of magnetization in every step for a temperature "b" 

    @Parameters
    -----------
    ->  b : float 
        ... beta parameter for the simulation
    ->  L : int
        ... length of the particles lattice
    ->  N : int
        ... sample size to compute the monte carlo estimator
    ->  init_cold : bool
        ... if true, the sistem will start in a cold state (every spins up (1))
    ->  vb : bool
        ... verbose argument
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

    # the monte carlo estimator for the sistema magnetization is computed
    M_mc = sum(M[20:])/len(M[20:]) # from the index 20 to have in mind the Ntherm

    print("%.3f \t %.3f"%(b, M_mc))
    return (b, M_mc)

def main(L, N, cold=True, verbose=True):   
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

    filename = "transitionp_L%d.txt"%L
    
    if verbose: print("runing several betas in parallel...")

    with open(filename, 'w') as file: # file to save the simulations times
        file.write("b \t M\n")
        
        pool = Pool(cpu_count())
        # the parallel simulation processes are created
        results = [pool.apply_async(run_simulation, args=[b, L, N, cold, verbose]) for b in betas]

        # the monte carlo estimators of magnetization are saved in a file
        for r in results:
            b , M = r.get()
            file.write("%.3f \t %.3f\n"%(b, M))


if __name__ == "__main__":
    #---------------------------- Args configurations ------------------------------      
    parser = argparse.ArgumentParser(description="This program runs the 2D ising\
    model simulation for differents beta values in parallel, it generates the \
    files with the data for compute the phase transition")

    parser.add_argument('-L', type=int, metavar='', default=10,
                    help='Lattice size')
    parser.add_argument('-N', type=int, metavar='', default=1000, 
                    help='number of updates of the lattice')
    parser.add_argument('-cold', type=int, metavar='', default=1, choices=[0,1],
                    help='if 1, init the lattice in cold state')
    parser.add_argument('-nv', type=int, metavar='', default=0, choices=[0,1],
                    help='verbose disable')

    args = parser.parse_args()
    # main execution...........
    main(args.L, args.N, args.cold, not args.nv)