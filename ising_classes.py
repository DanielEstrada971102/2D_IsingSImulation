""" ****************************************************************************
* @file ising_classes.py                                                       *
* @author Daniel Estrada (daniel.estrada1@udea.edu.co)                         *
* @brief 2D Ising model classes definition                                     *
* @version 0.2                                                                 *
* @date 2021-11-06                                                             *                                     
*                                                                              *
* @copyright Copyright (c) 2021                                                *
********************************************************************************
"""
from numpy import random, exp, array, reshape
from copy import deepcopy
import argparse
from itertools import product
from tqdm import tqdm

#-------------------------------- Part Class -----------------------------------
class Part():
    """
    This class represents a particle in the 2D ising lattice. 

    @parameters
    -----------
    ->  pos : int list or int tuple
        ... tuple with the (i, j) position of the particle into the lattice
    ->  s_value : int
        ... initial spin value
    ->  L : int
        ... length of the particles lattice

    @Attributes 
    -----------
    ->  spinValue : int
        ... 1 if the particle's spin is up, -1 if it is down.
    ->  energyValue : int
        ... particle energy value, it depends of the system configuration
    ->  neighbors : list
        ... list with the position of each neighbor of particle, continuos 
            boundary conditions are considered

    """
    def __init__(self, pos, s_value, L):
        # initializing  attributes
        self.spinValue = s_value
        self.energyValue = 0
        
        # setting the positions of their neighbors 
        # modul operator is for guarantee periodic boundary conditions
        i , j = pos
        self.neighbours = [ (i%L, (j-1)%L),        # left
                            (i%L, (j + 1)%L),      # right
                            ((i-1)%L, j%L),        # top
                            ((i+1)%L, j%L)]        # bottom
    
#------------------------------ IsingLattice Class -----------------------------
class IsingLattice():
    """
    This class represents a square lattice of particles in the 2D ising model. 

    @parameters
    -----------
    ->  L : int
        ... length of the particles lattice, the system will be a LxL matrix
    ->  beta : float
        ... represent the system temperature. 
    ->  init_cold : bool
        ... if true, the sistem will start in a cold state (every spins up (1))

    @Attributes 
    -----------
    ->  beta : float
        ... system temperature paramter
    ->  L : int
        ... length of the particles lattice
    ->  M : float
        ... total system magnetization
    ->  E : int
        ... total system energy
    ->  S : array of Parts
        ... LxL array, each entrance is a instance of Part.
    """

    def __init__(self, L = 10, beta = 0.2, init_cold = True):
        # initializing attributes
        self.beta = beta 
        self.L = L
        self.M = 0 
        self.E = 0 

        # setting macroscopic quantities and the initial spins configuration
        # -- the system begins in a could or hot configuration
        self.init_lattice(init_cold)


    def init_lattice(self, cold):
        """
        this function sets the initial spin configuration, energy and 
        magnetization of the system. 

        @Parameters
        -----------
        ->  cold : bool
            ... if true, all spin values will be up (1) (cold configuration),
                else, spin will be randomly set (+-1) (hot configuration)
        @Retuns
        -------
        -
        """
        if cold:
            self.S = array([Part((i,j), 1, self.L) for i, j in 
                            product(range(self.L), range(self.L))])
        else:
            self.S = array([Part((i,j), random.choice([-1,1]), self.L) 
                            for i, j in product(range(self.L), range(self.L))])

        # the individual energy value for each particle is computed
        for part in self.S:
            self.update_individualEnergy(part)

        # the initial energy of the system is the sum...
        self.E = sum([part.energyValue for part in self.S]) 
        # the initial magnetization is the average of the particles spins
        self.M = sum([part.spinValue for part in self.S]) / self.L ** 2

    def supIndex(self, i, j):
        return j + i * self.L

    def update_individualEnergy(self, part):
        """
        computes the energy of Part according to the neighbors configuration.

        @Parameters
        -----------
        ->  part : Part
            ... particle whom the enegy gonna be computed
        @Retuns
        -------
        -
        """
        sum = 0

        for i,j in part.neighbours:
            index = self.supIndex(i, j)    
            sum += self.S[index].spinValue

        part.energyValue = -0.5 * part.spinValue * sum

    def p_acceptance(self, dH):
        """
        computes the acceptance probabilty used in the metropolis-hasting 
        algorithm. 

        @Parameters
        -----------
        ->  dH: int
            ... change in energy
        @Retuns
        -------
        ->  float: probability of accept the system spin configuration change
        """
        return exp( -1 * self.beta * dH)

    def updateLattice(self, N=1, verbose = False):    
        """
        run N steps of the Metropilis-Hasting algorithm. L**2 runs represents
        one update of the spin system configuration. 

        @Parameters
        -----------
        ->  N: int
            ... number of algorithm repetitions
        ->  verbose : bool
            ... if true, a progress bar is showed...
        @Retuns
        -------
        -
        """
        for n in tqdm(range(N), disable= not verbose):
            change = False

            # a random position in the lattice is chosen
            index = random.randint(0, self.L**2) 
            S_aux = deepcopy(self.S[index])
            # the spin is flipped
            S_aux.spinValue = -1 * S_aux.spinValue

            # the resulting change in energy is computed
            self.update_individualEnergy(S_aux)
            dE =  S_aux.energyValue - self.S[index].energyValue

            if dE >= 0 :
                if random.uniform() >= self.p_acceptance(dE): 
                    change = False  
                else: 
                    change = True 
            else: # always accept the change if the new energy is smaller
                change = True 
            
            if change: #if the change was accepted update the system
                self.S[index] = deepcopy(S_aux)
                self.E += dE
                self.M += self.S[index].spinValue / self.L ** 2


    def get_spinMatrix(self):
        """
        extracts the system spin values and put them in a matrix

        @Parameters
        -----------
        -
        @Retuns
        -------
        ->  numpy array : matrix with the system spin values
        """
        spinValues = [part.spinValue for part in self.S]
        
        return reshape(spinValues, (self.L,self.L))

#---------------------------------- test ---------------------------------------      
def main(L, b, cold, m):
    """"
    Test of the classes funcionality
    """
    test_System = IsingLattice(L, b, cold)

    print(test_System.get_spinMatrix())
    print("Total M: %.4f"%test_System.M)
    print("Total E: %.4f"%test_System.E)

    test_System.updateLattice(m, verbose=True)

    print(test_System.get_spinMatrix())
    print("Total M: %.4f"%test_System.M)
    print("Total E: %.4f"%test_System.E)

    
if __name__ == "__main__":
    #---------------------------- Args configurations ------------------------------      
    parser = argparse.ArgumentParser(description="Here are defined the 2D Ising \
    model class, the main function run a test of the classes funtionality")
    parser.add_argument('-L', type=int, metavar='', default=10,
                    help='lenght of the lattice')
    parser.add_argument('-cold', type=int, metavar='', default=1, 
                    help='init the lattice in cold state')
    parser.add_argument('-m', type=int, metavar='', default=1, 
                    help='number of repetition to update the lattice')
    parser.add_argument('-b', type=float, metavar='', default=.2, 
                    help='temperature parameter b = 1/kT')
    args = parser.parse_args()

    main(args.L, args.b, args.cold, args.m)

