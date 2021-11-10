""" ****************************************************************************
* @file get_outputs.py                                                         *
* @author Daniel Estrada (daniel.estrada1@udea.edu.co)                         *
* @brief - takes the data files genereted by run.py and computes the monte     *
*          carlo estimator <M> for each beta in [bMin:bMaxto:bSteps]           *
*        - graphs the phase transicion with <M>, Xu and C, and gets the beta_c *
*        - produces a gif of the spins configurations.                         *
* @version 0.1                                                                 *
* @date 2021-10-14                                                             *                                     
*                                                                              *
* @copyright Copyright (c) 2021                                                *
********************************************************************************
"""
from os import remove
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
from numpy import loadtxt, array, arange, mean, log, sqrt, diff, where
from scipy import interpolate
from multiprocessing import Process, Queue


def thermalization(b, dir, save=False, Nskip = 1, parallel = (False, None)):
    """
    This function does a thermalization graph for beta = b, and return an Ntherm 

    @Parameters
    -----------
    ->  b : float 
        ... temperature parameter of the system
    ->  dir : str
        ... directory where data are
    ->  save : bool
        ... if True the graphs will be saved in "dir"
    ->  Nskip : int
        ... grahps the data only every Nskip steps
    @Retuns
    -------
    ->  int : 
    """
    print("building the thermalization graph...")
    datafile = dir + "/data/b%.3f.txt"%b
    M, _ = loadtxt(datafile, skiprows=1, unpack=True)

    # an estimation of Ntherm is computed 
    Ntherm = where(diff(M) == max(diff(M)))[0][0] + 10
    print("An acceptable value for Ntherm can be %d"%Ntherm)
    
    # the termalization graph is made
    fig, ax = plt.subplots()
    fig.figsize=(8,10)

    ax.plot(M[::Nskip], '.')
    ax.set_title(r"Magnetización del sistema, $b = %.3f$"%b +"\n", fontsize=15)
    ax.set_ylabel(r"$M_n$", fontsize=12)
    ax.set_xlabel(r"$n$", fontsize=12)
    ax.grid(True)
    
    if save:
        image_dir = dir + "/termalizacion_b%.3f.png"%b
        fig.savefig(image_dir)
        plt.close()
    else:
        plt.show()
    
    if parallel[0]:
        parallel[1].put(Ntherm)
    else:
        return Ntherm

def phase_Transition(dir, betas, Nskip=1, Ntherm=0, save=False):
    """
    This function does the transition graphs ignoring the first Ntherm terms and
    taking the data every Nskip for "betas" values. with the data, the function 
    computes an estimation for b_c.

    @Parameters
    -----------
    ->  dir : str
        ... directory where data are
    ->  betas : float list or array/arange : numpy
        ... list with the beta variation for the simulation
    ->  Ntherm : int
        ... grahps the data from Ntherm-th date.
    ->  Nskip : int
        ... grahps the data only every Nskip steps
    ->  save : bool
        ... if True the graphs will be saved in "dir"
    @Retuns
    -------
    -
    """
    # extracting the length of the lattice from the folder name
    L = int(dir.partition('_')[-1])

    Mprom = []
    M2prom = []
    Eprom = []
    E2prom = []
    
    print("computing the monte carlo estimators for every beta...")
    for b in tqdm(betas):
        # the data of the system observables are loaded for every beta
        dataFile = dir + "/data/b%.3f.txt"%b
        M, E = loadtxt(dataFile, skiprows=1, unpack=True)
        
        # the montecarlo estimators are computed
        Mprom.append( mean( M[Ntherm::Nskip] ) )
        M2prom.append(mean(M[Ntherm::Nskip]**2))
        Eprom.append(mean(E[Ntherm::Nskip]))
        E2prom.append(mean(E[Ntherm::Nskip]**2))

    # the lists are changed to numpy arrays to do operations with the data
    Mprom = array(Mprom)
    M2prom = array(M2prom)
    Eprom = array(Eprom)
    E2prom = array(E2prom)
    # specific heat
    C = betas**2 * (E2prom - Eprom**2) * 1/L**2
    # magnetic susceptibility
    Xu = betas * (M2prom - Mprom**2) * L**2

    print("smoothing the data...")
    # the data are smothed to beauty the graphs
    MpromS = smooth(betas, betas, Mprom)
    Es = smooth(betas, betas, Eprom)
    Cs = smooth(betas, betas, C)
    XuS = smooth(betas, betas, Xu)

    # the critical b is computed as the average between the 
    # critical value of every observable
    bc1 = compute_bc(betas, Mprom)
    bc2 = compute_bc(betas, Xu, Max=True)
    bc3 = compute_bc(betas, C, Max=True)
    
    bc = 1/3 * (bc1 + bc2 + bc3)
    bc_t = 0.5 * log(1 + sqrt(2))
    err_bc = abs((bc - bc_t)/bc_t) 
    
    print("The critial value of beta is : %.3f +- %.3f"%(bc,err_bc))

    print("bilding the transition graphs...")
    #.................. the phase transition graphs are made ................
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,sharex=True, figsize=(15,8))
    
    ax1.plot(betas, Eprom, '.r')
    ax1.plot(betas, Es, '--r')
    ax1.set_ylabel(r"$\left<E\right>$", fontsize=15)
    ax1.grid(True)
    
    ax2.plot(betas, Mprom, '.b')
    ax2.plot(betas, MpromS, '--b')
    ax2.axvline(bc, ymin=0, ymax=1, color='k', linestyle='--', alpha=0.8)
    ax2.set_ylabel(r"$\left<M\right>$", fontsize=15)
    ax2.grid(True)

    ax3.plot(betas, Xu, '.g')
    ax3.plot(betas, XuS, '--g')
    ax3.axvline(bc, ymin=0, ymax=1, color='k', linestyle='--', alpha=0.8)
    ax3.set_ylabel(r"$\chi_{\mu}$", fontsize=15)
    ax3.set_xlabel(r"$\hat{\beta}$", fontsize=15)
    ax3.grid(True)
    
    ax4.plot(betas, C, '.', color='orange')
    ax4.plot(betas, Cs, '--', color='orange')
    ax4.axvline(bc, ymin=0, ymax=1, color='k', linestyle='--', alpha=0.8)
    ax4.set_ylabel(r"$C$", fontsize=15)
    ax4.set_xlabel(r"$\hat{\beta}$", fontsize=15)
    ax4.grid(True)
    
    
    fig.suptitle("\nTransición de fase L=%d\n"%L, fontsize=20)

    if save:
        image_dir = dir + "/transicion.png"
        fig.savefig(image_dir)
        plt.close()
    else:
        plt.show()
    

def compute_bc(betas, data, Max=False):
    """
    This function compute a critical value from "betas" according to "data", it 
    is possible to do the process with the max diff in data or with the max value simply 

    @Parameters
    -----------
    ->  betas : float list/array
        ... list with the beta values
    ->  data :  float list/array 
        ... list with the date of the system observables
    ->  Max : bool
        ... if true, the critical parameter is computed like the beta whom 
            do data maximum
    @Retuns
    -------
    ->  float : the critial value 
    """
    try:
        if Max:
            indx = where(data == max(data))[0][0]    
        else:
            indx = where(diff(data) == max(diff(data)))[0][0]
        
        bc = betas[indx]
    
    except:
        bc = 0
    return bc

def smooth(x, X, Y): #Función para suavizar los datos
    """
        This function is only for smooth the data and do a more estetic graph
    """
    f = interpolate.interp1d(X,Y, kind = 'cubic')
    y = f(x)
    return y


def print_System(dir, b, N, Nskip=1, Ntherm=0):
    """
    This function generates the spin configuration images to create a gif that 
    shows the system evolution. 

    @Parameters
    -----------
    ->  dir : str
        ... directory where spin configuration data are
    ->  b : float 
        ... temperature parameter of the system
    ->  N : int
        ... total simulation steps
    ->  Nskip : int
        ... takes the data only every Nskip configurations
    ->  Ntherm : int
        ... ignores the first Ntherm configurations.
    @Retuns
    -------
    -
    """
    # the path  of the spin configutarions file is defined
    configFiles = dir + "/visual/b_%.3f"%b
    # extracting the length of the lattice from the folder name
    L = dir.partition('_')[-1]

    imageNames = [] #the image names are necesary to do the gif
    
    print("bilding the system spin configuration images..")
    
    for n in tqdm(range(Ntherm, N, Nskip)):
        system = loadtxt(configFiles +"/step%d.txt"%n)
        
        fig, ax = plt.subplots(figsize=(9,7))
        ax.matshow(system, cmap = plt.cm.plasma) 
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title('Sistema de espines, L=%s b=%.3f step=%d'%(L ,b,n), fontsize = 15)
        filename = dir + "/step%d.png"%n
        imageNames.append(filename)
        plt.savefig(filename)
        plt.close()

    # buildint the gif
    with imageio.get_writer(dir + '/L%s_b%.3f.gif'%(L,b), mode='I') as gif:
        for filename in imageNames:
            image = imageio.imread(filename)
            gif.append_data(image)
            
    # Removing geneated images
    for filename in set(imageNames):
        remove(filename)    


def main(L, attmp,bmin, bmax, bstep, N, Nskip, save, gen_img, parallel=False):
    dir = "L%s_%d"%(attmp, L) 

    betas = arange(bmin, bmax, bstep)    

    if parallel:
        # queue = Queue()

        # proc1 = Process(target=thermalization, 
        #                 args=[0.4, dir, save, 1, (parallel, queue)]).start()
        # proc2 = Process(target=thermalization, 
        #                 args=[0.8, dir, save, 1, (parallel, queue)]).start()
        
        # proc1.join()
        # proc2.join()
        
        #Ntherm = int(0.5 * (queue.get() + queue.get()))
        phase_Transition(dir, betas, Nskip, save=save)
        
        if gen_img:
            proc3 = Process(target=print_System, 
                            args=[dir, 0.4, N, Nskip]).start()
            proc4 = Process(target=print_System, 
                            args=[dir, 0.8, N, Nskip]).start()
            proc3.join()
            proc4.join()

    else:
        #Ntherm1 = thermalization(0.4, dir, save)
        #Ntherm2 = thermalization(0.8, dir, save)
        #Ntherm = int(0.5 * (Ntherm1 + Ntherm2))

        phase_Transition(dir, betas, Nskip, save=save)
        
        if gen_img:
            print_System(dir, 0.4, N, Nskip)
            print_System(dir, 0.8, N, Nskip)
    

if __name__ == "__main__":
    #---------------------------- Args configurations ------------------------------      
    parser = argparse.ArgumentParser(description="This program takes the data \
    files genereted by run.py and computes the monte carlo estimator <M> for \
    every beta in [bmin:bmax:bsteps],then, it graphs the phase transicion with\
    <M>, Xu and C, and gets the beta_c, it's posible produce a gif of the spins \
    configurations too.")
    parser.add_argument('-L', type=int, metavar='', default=10, 
                    help='identificator of data folder')
    parser.add_argument('-attmp', type=str, metavar='', default='', 
                    help='If you have more than once run for this L, indicates \
                          the attempt identificator')
    parser.add_argument('-bmin', type=float, metavar='', default=0,
                    help='min beta to analyze')
    parser.add_argument('-bmax', type=float, metavar='', default=1, 
                    help='max beta to analyze')
    parser.add_argument('-bstep', type=float, metavar='', default=.02, 
                    help='beta step')
    parser.add_argument('-N', type=int, metavar='', default=1000, 
                    help='number of data in the files')
    parser.add_argument('-Ns', type=int, metavar='', default=100, 
                    help='use the data only every Nskip ')
    parser.add_argument('-save', type=int, metavar='', default=0, choices=[0,1], 
                    help='if 1, the graphs will be saved')
    parser.add_argument('-gif', type=int, metavar='', default=0, choices=[0,1],
                    help='if 1, a giff of the system evolution is generated')
    parser.add_argument('-p', type=int, metavar='', default=0, choices=[0,1],
                    help='if 1, the process will develop in parallel')
   
    args = parser.parse_args()

    main(args.L, args.attmp, args.bmin, args.bmax, args.bstep, args.N, 
        args.Ns, args.save, args.gif, args.p)