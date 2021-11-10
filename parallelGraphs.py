import matplotlib.pyplot as plt
from numpy import loadtxt, std, sqrt, sinh
from scipy import interpolate, optimize
from multiprocessing import Pool, cpu_count


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
    plt.plot(l, Ts_s, '--r', label=r"$%.2f + %.2f x + \frac{%.2f}{2} x^2$"%(popt_s[0], popt_s[1],popt_s[2])) 
    plt.plot(L, Tp, 'ob')
    plt.plot(l, Tp_s, '--b', label=r"$%.2f + %.2f x + \frac{%.2f}{2} x^2$"%(popt_p[0], popt_p[1],popt_p[2]))
    plt.xlabel(r"$L$")
    plt.ylabel(r"$t$" + "  " + "(h)")
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()

    # pool = Pool(cpu_count())

    # results = [pool.apply_async(compute_magnetization, args=[l]) for l in L]

    # pool.close()
    # pool.join()

    # beta = 0.3
    # M_t = (1 - sinh(2*beta)) ** (1/8)

    # print(M_t)
    # for r in results : print(r.get())



def smooth(x, X, Y): #Función para suavizar los datos
    """
        This function is only for smooth the data and do a more estetic graph
    """
    # f = interpolate.interp1d(X,Y, kind = 'cubic')
    g = lambda x, a, b, c : a + b * x + c * 0.5 * x * x
    popt, pcov = optimize.curve_fit(g, X, Y)
    f = lambda x: g(x, *popt)

    y = f(x)
    return y, (popt, pcov)

# def compute_magnetization(L):
#     M, _ =loadtxt("data/b0.300_L_%d.txt"%L, unpack=True, skiprows=3)
#     M_mc = sum(M) / len(M)
#     err_M = std(M) / sqrt(len(M))

#     return M_mc, err_M

if __name__=='__main__':
    main()