""" ****************************************************************************
* @file run_times.py                                                           *
* @author Daniel Estrada (daniel.estrada1@udea.edu.co)                         *
* @brief Runs the 2D ising  model simulation, generates the files with the     *
*       data for compute magnetization and the save the computing time expended*
* @version 0.2                                                                 *
* @date 2021-11-8                                                             *                                     
*                                                                              *
* @copyright Copyright (c) 2021                                                *
********************************************************************************
"""
from time import perf_counter
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from ising_classes import * # importing the classes for the simulation


def run_simulation(b, L, N, m, init_cold = True, vb=True):
    """
    This function generates N updates of a LxL system for extract the values 
    of magnetization in every step for a temperature "b" and 
    saves the data in a file. 

    @Parameters
    -----------
    ->  b : float 
        ... beta parameter for the simulation
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
    @Retuns
    -------
    -
    """

    if vb : print("runing for L = %d and b = %.3f"%(L, b))

    #path to save the magnetization and energy data
    dataFile_name = "./b%.3f_L_%d.txt"%(b, L)

    with open(dataFile_name, 'w') as f:
        
        f.write("M\tE\n") #file header 
        
        system = IsingLattice(L, b, init_cold) # the particles sysyem is initialized
        
        # initial magnetization and energy values are written
        f.write("%.4f\t%.4f\n"%(system.M, system.E)) 
        # initial spins configuration is saved

        for n in tqdm(range(1, N), disable = not vb):         
            # a new spins configuration is genereted with Metropolis algorithm          
            system.updateLattice(L*L)            
            # new magnetization and energy values are written
            f.write("%.4f\t%.4f\n"%(system.M, system.E))


def updating_process(system, m):
    """
    helper function to do the process parallely
    
    @Parameters
    -----------
    ->  system : IsingLattice instance
        ... the 2d ising system
    ->  m : int
        ... number of updates to perform
    @Retuns
    -------
    --> tuple : (float, int) the value of Energy and Magnetization after m update
    """
    system.updateLattice(m)

    return (system. M, system.E)
    

def main(b, N, cold=True, verbose=True, parallel=False):   
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
    ->  parallel : bool
        ... if true, the simulation will be performed in parallel
    @Retuns
    -------
    -
    """

    sizes = [15, 30, 60, 120, 240, 480]
    
    filename = "tiempos_par.txt" if parallel else "tiempos_sec.txt"
    
    if parallel and verbose: print("runing in parallel...")

    with open(filename, 'w') as file: # file to save the simulations times
        file.write("L \t time(s)\n")
        
        for l in sizes:
            t_star = perf_counter()

            if verbose : print("runing for L = %d and b = %.3f"%(l, b))
            dataFile_name = "./%sb%.3f_L_%d.txt"%("p" if parallel else "s", b, l)

            with open(dataFile_name, 'w') as f:
                f.write("M\tE\n")
                system = IsingLattice(l, b)
                f.write("%.4f\t%.4f\n"%(system.M, system.E)) 
                
                # ================== run simualition in Parallel =======================
                if parallel:
                    pool = Pool(cpu_count())
                    results = []
                    for _ in tqdm(range(1, N), disable = not verbose):
                        results.append(pool.apply_async(updating_process, args=[system, l*l]))
                    
                    pool.close()
                    pool.join()
                   
                    for r in results:
                        M, E = r.get()
                        f.write("%.4f\t%.4f\n"%(M, E))

                # ================== run simualition sequentialy ===============
                else:
                    run_simulation(b, l, N, cold, verbose)

            t_end = perf_counter()
            if verbose : print(" time spended : ", t_end - t_star, " s")
            file.write("%d \t %.5f\n"%(l, t_end - t_star))


if __name__ == "__main__":
    #---------------------------- Args configurations ------------------------------      
    parser = argparse.ArgumentParser(description="This program runs the 2D ising\
    model simulation, generates the files with the data for compute magnetization\
    the main function executs the simulation with the parameters passed.")

    parser.add_argument('-b', type=float, metavar='', default=0.3,
                    help='temperature parameter')
    parser.add_argument('-N', type=int, metavar='', default=1000, 
                    help='number of updates of the lattice')
    parser.add_argument('-cold', type=int, metavar='', default=1, choices=[0,1],
                    help='if 1, init the lattice in cold state')
    parser.add_argument('-nv', type=int, metavar='', default=0, choices=[0,1],
                    help='verbose disable')
    parser.add_argument('-p', type=int, metavar='', default=0, choices=[0,1],
                    help='if 1, run in parallel')

    args = parser.parse_args()
    # main execution...........
    main(args.b, args.N, args.cold, not args.nv, args.p)