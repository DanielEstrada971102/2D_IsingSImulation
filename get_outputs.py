""" ****************************************************************************
* @file Graphs.py                                                              *
* @author Daniel Estrada (daniel.estrada1@udea.edu.co)                         *
* @brief code to process the time simualion data generated                     *
* @version 0.2                                                                 *
* @date 2021-11-09                                                             *                                     
*                                                                              *
* @copyright Copyright (c) 2021                                                *
********************************************************************************
"""
import matplotlib.pyplot as plt
from numpy import loadtxt, std, where, diff
from scipy import optimize, interpolate

def graphic_times():
    """
    This function produces the graph of the execution time and computes the 
    observed speedup value.
    
    @Parameters
    -----------
    -
    @Retuns
    -------
    -
    """
    # the time data files is loaded
    L, Ts = loadtxt("data/tiempos_sec.txt", unpack=True, skiprows=1)
    _, Tp = loadtxt("data/tiempos_par.txt", unpack=True, skiprows=1)

    # change from seconds to hours
    Ts = Ts * 1/(60*60)
    Tp = Tp * 1/(60*60)

    # the data are smoothed
    l = range(1, 500)
    Ts_s, (popt_s, pcov_s) = smooth(l, L, Ts)
    Tp_s, (popt_p, pcov_p) = smooth(l, L, Tp)
    
    # the observed speedup value is computed
    spUP = sum(Ts/Tp) / len(Ts)
    err_spUP = std(Ts/Tp)

    print("Obs.Speedup is ",spUP, "+-", err_spUP)

    # the graph is produced
    fig, ax = plt.subplots()
    ax.plot(L, Ts, 'or')
    ax.plot(l, Ts_s, '--r', label="Serial")#, label=r"$%.2e + %.2e x^2$"%(popt_s[0], popt_s[1])) 
    ax.plot(L, Tp, 'ob')
    ax.plot(l, Tp_s, '--b', label="Paralelo")#, label=r"$%.2e + %.2e x^2$"%(popt_p[0], popt_p[1]))
    ax.set_title(r"Tiempo de ejeción $vs$ L", fontsize=15)
    ax.set_xlabel(r"$L$", fontsize=12)
    ax.set_ylabel(r"$t$" + "  " + "(h)", fontsize=12)
    ax.grid(True)
    ax.legend(loc='best')
    fig.savefig("RunTime.png")
    plt.close()

    

def compute_bc(L):
    """
    This function computes an estimation of critical parameter for a L size 
    system
    
    @Parameters
    -----------
    -->  L : int
         ... Lattice size
    @Retuns
    -------
    -
    """
    b , M = loadtxt("data/transitionp_L%d.txt"%L, unpack=True, skiprows=1)

    # the critical value is computed as the point where the diff increase
    indx = where(diff(M) > 0.1)[0][0]  
    bc =  b[indx]

    print("for L=", L, " b_c is ", bc)

    f = interpolate.interp1d(b,M, kind = 'cubic')
    Ms = f(b)


    # the graph is produced
    fig, ax = plt.subplots()
    ax.plot(b, M, 'ob')
    ax.plot(b, Ms, '--k')
    ax.set_title(r"Transición de fase para L = %d"%L, fontsize=15)
    ax.set_xlabel(r"$\beta$", fontsize=12)
    ax.set_ylabel(r"$<M>$", fontsize=12)
    ax.grid(True)
    fig.savefig("transicion_L%d.png"%L)
    plt.close()


def smooth(x, X, Y): #Función para suavizar los datos
    """
        This function is only for smooth the data and adjust its to a polynomial 
        of degree 3
    """
    g = lambda x, a, b : a + b * x * x
    popt, pcov = optimize.curve_fit(g, X, Y)
    f = lambda x: g(x, *popt)
    y = f(x)
    return y, (popt, pcov)

def main():
    # ================== to grapghic the time improvement ======================
    graphic_times()    
    
    # ===================== to grapghic phase trasition ========================
    compute_bc(10)
    compute_bc(30)


if __name__=='__main__':
    main()