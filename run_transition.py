from multiprocessing import  Pool
from ising_classes import * # importing the classes for the simulation
from numpy import arange


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
    --> float : the value of Magnetization after m update
    """
    system.updateLattice(m)

    return system. M

def main(np = 10):
    """
    This function runs Ising simulations for betas in range(0.1, 1, 0.01), for a 
    lattice of size 256x256. A montecarlo estimate of magnetization with 1000 values  
    is computed with a Metropolis-Hassting processes of 256x256 steps of the algorithm
    for each beta. The data is saved in a file to graph the phase transition.  
    
    Note: the only way to run this program quickly is using the cluster, cause 
    for each beta the time expended is around 2h using 4 processor. 
    
    @Parameters
    -----------
    ->  n : int 
        ... number of processors for run in the cluster 
    @Retuns
    -------
    -
    """
    N = 1000 
    L = 256
    betas = arange(0.1, 1, 0.01)
    
    # file to save the simulations times
    with open("/scracth/gfif-user0/DanielE/transition.txt", 'w') as file: 
        file.write("b \t M\n")
        
        for b in betas:
            print("runing for beta = ", b, "...")

            system = IsingLattice(L, b)               
            # ================== run simualition in Parallel =======================
            pool = Pool(np) # for use only 10 processors in the cluster
            results = []
            
            for _ in range(1, N):
                results.append(pool.apply_async(updating_process, args=[system, L*L]))
            
            pool.close()
            pool.join()
            #................. computing the monte carlo estimator .............
            Ms = []
            for r in results:
                Ms.append( r.get())
            # writing in the file...
            file.write("%.3f \t %.3f\n"%(b, sum(Ms)/len(Ms)))


if __name__=='__main__':
     #---------------------------- Args configurations ------------------------------      
    parser = argparse.ArgumentParser(description="This function runs Ising \
        simulations for betas in range(0.1, 1, 0.01), for a lattice of size\
        256x256. A montecarlo estimate of magnetization with 1000 values is \
        computed with a Metropolis-Hassting processes of 256x256 steps of the\
        algorithm for each beta. The data is saved in a file to graph the phase transition.")

    parser.add_argument('-n', type=int, metavar='', default=10,
                    help='number of proccesor to use')

    args = parser.parse_args()
    main(args.n)