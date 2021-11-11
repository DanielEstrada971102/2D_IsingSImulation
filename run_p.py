""" ****************************************************************************
* @file run_p2.py                                                              *
* @author Daniel Estrada (daniel.estrada1@udea.edu.co)                         *
* @brief Runs (In parallel) the 2D ising  model simulation for several lattice *
*        size values, generates the files with the data for compute            *
*        magnetization and the save the computing time expended                *
* @version 0.2                                                                 *
* @date 2021-11-8                                                              *                                     
*                                                                              *
* @copyright Copyright (c) 2021                                                *
********************************************************************************
"""
from time import perf_counter
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from ising_classes import * # importing the classes for the simulation


def main(b, N, outfolder, cold=True, verbose=True):   
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
    
    filename = outfolder + "2tiempos_par.txt"
    
    if verbose: print("runing N metropolis in parallel for b = %.3f..."%b)

    with open(filename, 'w') as file: # file to save the simulations times
        file.write("L \t time(s)\n")
        
        for l in sizes:
            t_star = perf_counter()

            if verbose : print("runing L = %d"%(l))
           
            system = IsingLattice(l, b, cold)
            
            # ================== run simualition in Parallel ===============
            pool = Pool(cpu_count())
            for _ in tqdm(range(1, N), disable = not verbose):
                pool.apply_async(system.updateLattice, args=[l*l])
            
            pool.close()
            pool.join()
            
            #===============================================================
                
            t_end = perf_counter()
            if verbose : print(" time spended : ", t_end - t_star, " s")
            file.write("%d \t %.5f\n"%(l, t_end - t_star))


if __name__ == "__main__":
    #---------------------------- Args configurations ------------------------------      
    parser = argparse.ArgumentParser(description="This program runs the 2D ising\
    model simulation for differents lattice size in parallel, it generates the \
    files with the data for compute magnetization")

    parser.add_argument('-b', type=float, metavar='', default=0.3,
                    help='temperature parameter')
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
    main(args.b, args.N, args.outfolder, args.cold, not args.nv)