""" ****************************************************************************
* @file Graphs.py                                                              *
* @author Daniel Estrada (daniel.estrada1@udea.edu.co)                         *
* @brief code to process the simualion data generated                          *
* @version 0.1                                                                 *
* @date 2021-11-09                                                             *                                     
*                                                                              *
* @copyright Copyright (c) 2021                                                *
********************************************************************************
"""
import matplotlib.pyplot as plt
from numpy import loadtxt
from scipy import optimize


def main():

    # ================== fro grapghic the time ======================
    L, Ts = loadtxt("data/tiempos_sec.txt", unpack=True, skiprows=1)
    _, Tp = loadtxt("data/tiempos_par.txt", unpack=True, skiprows=1)

    # change from seconds to hours
    Ts = Ts * 1/(60*60)
    Tp = Tp * 1/(60*60)

    l = range(1, 500)
    Ts_s, (popt_s, pcov_s) = smooth(l, L, Ts)
    Tp_s, (popt_p, pcov_p) = smooth(l, L, Tp)
    

    plt.plot(L, Ts, 'or')
    plt.plot(l, Ts_s, '--r', label="Serial")#, label=r"$%.2e + %.2e x^2$"%(popt_s[0], popt_s[1])) 
    plt.plot(L, Tp, 'ob')
    plt.plot(l, Tp_s, '--b', label="Paralelo")#, label=r"$%.2e + %.2e x^2$"%(popt_p[0], popt_p[1]))
    plt.title(r"Tiempo de ejeción $vs$ L", fontsize=15)
    plt.xlabel(r"$L$", fontsize=12)
    plt.ylabel(r"$t$" + "  " + "(h)", fontsize=12)
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig("RunTime.png")
    plt.show()


def smooth(x, X, Y): #Función para suavizar los datos
    """
        This function is only for smooth the data and adjust its to a polynomial 
        of degree 3
    """
    # f = interpolate.interp1d(X,Y, kind = 'cubic')
    g = lambda x, a, b : a + b * x * x
    popt, pcov = optimize.curve_fit(g, X, Y)
    f = lambda x: g(x, *popt)
    print(popt)
    y = f(x)
    return y, (popt, pcov)


if __name__=='__main__':
    main()