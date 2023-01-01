import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit


V = 3

A1 = 2553
E1 = 34000

A2 = 2.1*10**9
E2 = 7.7 * 10**4

A3 = 8.4*10**9
E3 = 5.2*10**4

A4 = 9.45*10**4
E4 = 3.75*10**4

Q1 = 62540000
Q2 = 10**9
Q4 = 1.08*10**9

R = 8.31

K_t = 700
T_vh = 40+273
ro = 1180
C_t = 1200
T_t = 8+273

C1_vh = 0.55 
C2_vh = 0.35
C5_vh = 0.10
T_max = 60+273

@jit
def math_model(F, TT):
    result_1 = list()
    result_2 = list()
    result_3 = list()
    result_4 = list()
    result_5 = list()
    result_6 = list()
    T_h = list()
    Time = list()
    d_t = 0.3
    T_h.append(T_vh)
    result_1.append(C1_vh)
    result_2.append(C2_vh)
    result_3.append(0.0)
    result_4.append(0.0)
    result_5.append(C5_vh)
    result_6.append(0.0)
    Time.append(0)
    cnt = -1 

    k1 = lambda TT: A1 * np.exp((-E1) / (R * TT))
    k2 = lambda TT: A2 * np.exp((-E2) / (R * TT))
    k3 = lambda TT: A3 * np.exp((-E3) / (R * TT))
    k4 = lambda TT: A4 * np.exp((-E4) / (R * TT))

    d_C3 = lambda C1,C2,C3,C6,t : k1(t)*C1*C2-k4(t)*C3*C6
    d_C1 = lambda C1,C2,t: -k1(t)*C1*C2-k2(t)*C1*C2
    d_C2 = lambda C1,C2,C3,C6,t: -k1(t)*C1*C2-k2(t)*C1*C2+k4(t)*C3*C6
    d_C6 = lambda C4,C5,C3,C6,t: k3(t)*C4*C5-k4(t)*C3*C6
    d_C4 = lambda C1,C2,C4,C5,t: k2(t)*C1*C2-k3(t)*C4*C5
    d_C5 = lambda C3,C6,C4,C5,t: -k3(t)*C4*C5 + k4(t)*C3*C6
    d_T = lambda d_C1,d_C2,d_C3,d_C6,T,F: (k1(T)*d_C1*d_C2*Q1*V+k2(T)*d_C1*d_C2*Q2*V+k4(T)*d_C3*d_C6*Q4*V-K_t*F*(T-TT)) / (ro*V*C_t)
    
    while result_3[-1] <= 0.3:
        cnt += 1
        result_3.append(result_3[-1] + d_C3(result_1[-1], result_2[-1], result_3[cnt], result_6[-1], T_h[-1]) * d_t)
        result_1.append(result_1[-1] + d_C1(result_1[-1], result_2[-1], T_h[-1]) * d_t)
        result_2.append(result_2[-1] + d_C2(result_1[-2], result_2[-1], result_3[cnt], result_6[-1], T_h[-1]) * d_t)
        result_6.append(result_6[-1] + d_C6(result_4[-1], result_5[-1], result_3[cnt], result_6[-1], T_h[-1]) * d_t)
        result_4.append(result_4[-1] + d_C4(result_1[-2], result_2[-2], result_4[-1], result_5[-1], T_h[-1]) * d_t)
        result_5.append(result_5[-1] + d_C5(result_3[cnt], result_6[-2], result_4[-2], result_5[-1], T_h[-1]) * d_t)
        T_h.append(T_h[-1] + d_T(result_1[-2], result_2[-2], result_3[cnt],result_6[-2], T_h[-1], F) * d_t)
        Time.append(Time[-1] + d_t)
        
        if T_h[-1] > T_max:
            return Time[-1], F, T_h[-1], result_3[-1]

    return Time[-1], F, T_h[-1], result_3[-1]


def minimum():
    F_range = np.linspace(6.5, 10.5, 100)
    Time_list = list()
    F_list = list()
    Temp_list = list()
    C_list = list()
    for i in F_range:
        tempTime, tempF, tempT, tempC = math_model(i, T_t)
        Time_list.append(tempTime)
        F_list.append(tempF)
        Temp_list.append(tempT)
        C_list.append(tempC)
    min_time = min(el for el in Time_list if Temp_list[Time_list.index(el)]-273 < 60)
    minTime, minF = min_time, F_list[Time_list.index(min_time)]
    plot3D(F_list, Time_list, C_list)
    return minTime, minF

def imitation(Tt):
    time, numbers = lab_6()
    numbers.insert(0, Tt)
    minTime, minF = minimum()
    print(f'Минимальное время: {minTime}')
    print(f'Минимальная площадь: {minF}')
    List_C = list()
    for i in numbers:
        tempTime, tempF, tempT, tempC = math_model(minF, i+273)
        List_C.append(tempC)

    plot(time, numbers[:990],List_C[:990])

def lab_6():
    lam1 = 5 ** 12
    lam2 = 3 ** 11
    M0 = 15
    disp0 = 4
    alpha0 = -0.085
    N = 1000
    Ns = 10
    x = [1]
    xi = []
    for j in range(N):
        x.append(((lam1 * x[j]) % lam2))
        xi.append(x[j] / lam2 - 0.5)
    Mx = 1 / N * sum(xi)
    peremen = 0
    for j in range(N):
        peremen += (xi[j] - Mx) ** 2
    dispx = 1 / N * peremen
    z = []
    for k in range(1, N - 9):
        j = 0
        for i in np.arange(k, k + Ns):
            j += xi[i] * np.sqrt(disp0 / (dispx * -alpha0)) * np.exp(alpha0 * (i -k)) + M0
        z.append(1 / Ns * j)
    x1 = [i for i in range(N)]
    z1 = [i for i in range(N - 10)]
    return z1, z 

def plot(Time,y,C):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6), label='Coursework')
    axes[0].plot(Time, y, color='black')
    axes[0].grid(True)
    axes[0].set(xlabel="Время(с)", ylabel="")
    axes[0].axis(ymin=7,ymax=20)
    axes[1].plot(Time, C, color='black')
    axes[1].grid(True)
    axes[1].set(xlabel="Время(с)", ylabel="Концентрация(моль/м^3)")
    axes[1].axis(ymin=0.1,ymax=0.31)
    plt.subplots_adjust(wspace=0.5, hspace=0)
    plt.show()

def plot3D(F,t,C):
    fig = plt.figure(figsize=(10, 6))
    axes = Axes3D(fig, auto_add_to_figure=False)
    axes.plot_trisurf(F, t, C, cmap='plasma', edgecolor='none', antialiased=True)
    axes.set_xlabel('F')
    axes.set_ylabel('τ')
    axes.set_zlabel('C')
    fig.add_axes(axes)
    plt.show()

def main():
   imitation(8)

if __name__ == '__main__':
    main()